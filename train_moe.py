# coding=utf-8
from __future__ import absolute_import, division, print_function

import sys
sys.path.append(".")
import argparse

from datetime import timedelta

from torch.utils.tensorboard import SummaryWriter

from util.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from datasets.data_utils import get_loader
# from ViT.utils.utils import *
# from memory_profiler_backrazor import profile_memory_cost
from timm.models.vision_transformer import Attention, Mlp
from models.moe import MainAttention
import time
# import mesa as ms
import random
import torch
import torch.distributed as dist
from torch import nn
import os
import numpy as np
import models_vit
from models.moe import get_model
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import fmoe
from util.moe_utils import sync_weights
from models.moe import ViTMoE
from itertools import zip_longest
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def valid(args, model, writer, test_loader, global_step, log, task_list, phase="Validation"):
    # Validation!
    eval_losses = {task: AverageMeter() for task in task_list}
    # eval_losses = AverageMeter()

    log.info("***** Running {} *****".format(phase))
    log.info("  Num steps = {}".format(len(test_loader)))
    log.info("  Batch size = {}".format(args.eval_batch_size))

    end = time.time()

    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss()

    loss_dict, accuracy_dict = {}, {}
    for task_id, task in enumerate(task_list):
        all_preds, all_label = [], []
        for step, batch in enumerate(test_loader[task_id]):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            with torch.no_grad():
                logits = model(x, task_id)

                eval_loss = loss_fct(logits, y)
                eval_losses[task].update(eval_loss.item())

                preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            if step % 50 == 0:
                loss_string = "".join(["{}: {:.4f}\t".format(k, v.avg) for k, v in eval_losses.items()])
                log.info("Task:{} Validating {}/{} (loss={})".format(task, step, len(test_loader), loss_string))

        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        loss_dict[task] = eval_losses[task].avg
        accuracy_dict[task] = accuracy
    log.info("\n")
    log.info("Validation Results")
    log.info("Global Steps: {}".format(global_step))
    log.info("Valid Loss: {}".format(loss_dict))
    log.info("Valid Accuracy: {}".format(accuracy_dict))
    log.info("Time spent: {:.2f}".format(time.time() - end))

    if args.local_rank in [-1, 0]:
        writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy_dict

def save_model(args, model, log):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(log.path, "checkpoint_best.pth")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    log.info("Saved model checkpoint to [DIR: {}]".format(os.path.join(log.path, args.name)))

def train(args, model, train_loader, val_loader, task_list, log, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    memory_meter = AverageMeter()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_loader_iterator = [iter(loader) for loader in train_loader]
    # Prepare optimizer and scheduler
    if args.HeadLr10times:
        head_params = []
        feature_params = []
        for name, parameter in model.named_parameters():
            if "resnet" in args.model_type:
                head_name = "fc"
            else:
                head_name = "head"

            if head_name in name:
                head_params.append(parameter)
            else:
                feature_params.append(parameter)

        params_list = [{"params": filter(lambda p: p.requires_grad, feature_params)},
                       {"params": filter(lambda p: p.requires_grad, head_params), "lr": args.learning_rate * 10}]
    else:
        params_list = model.parameters()

    optimizer = torch.optim.SGD(params_list,
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    loss_scaler = NativeScaler()
    if args.distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    args.output_dir = os.path.join(args.output_dir, "checkpoints", args.name)
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    global_step = args.start_epoch # We store the gobal step as epoch in the saved checkpoint
    if global_step > 0:
        log.info("Resuming training from step {}".format(global_step))
        last_epoch = global_step
    else:
        last_epoch = -1
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total, last_epoch=last_epoch)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total, last_epoch=last_epoch)

    # Distributed training
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # global_step = 2900
    # Train!
    log.info("***** Running training *****")
    log.info("  Total optimization steps = {}".format(args.num_steps))
    log.info("  Instantaneous batch size per GPU = {}".format(args.train_batch_size))
    log.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)))
    log.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    loss_fct = torch.nn.CrossEntropyLoss()
    losses = {task: AverageMeter() for task in task_list}
    losses['total'] = AverageMeter()

    epoch = 0
    accuracy = -1
    best_acc = -1
    while True:
        log.info("set epochs: {}".format(epoch))
        if args.local_rank != -1:
            for train_loader_ in train_loader:
                train_loader_.sampler.set_epoch(epoch)
        epoch += 1

        model.train()
        end = time.time()
        # n_steps = 0
        # for task_id, task in enumerate(task_list):
        #     n_steps = max(n_steps, len(train_loader[task_id]))
                          
        for step, batches in enumerate(zip_longest(*train_loader)):
            data_time.update(time.time() - end)

            loss_dict = {}
            total_loss = 0
            # for task_id, task in enumerate(task_list):
            for task_id, batch in enumerate(batches):
                task = task_list[task_id]
                if batch is None:
                    continue
                # batch = next(train_loader_iterator[task_id])
                batch = tuple(t.to(args.device) for t in batch)

                x, y = batch

                # set_trace()
                # if "resnet" in args.model_type:
                pred = model(x, task_id)
                loss_ = loss_fct(pred, y)
                # loss_ = torch.sum(pred)
                loss_dict[task] = loss_.item()
                # TODO: add loss weight for different tasks
                total_loss += loss_.item() # accumulate loss for all tasks

                # update after every task
                loss_.backward()
                if isinstance(model.module.backbone, ViTMoE) and (not args.moe_data_distributed):
                    model.allreduce_params()
                optimizer.step()
                optimizer.zero_grad()
                # MB = 1024 * 1024
                # memory_meter.update(torch.cuda.max_memory_allocated() / MB)
                # log.info("Training ({}/{} Steps)\tData time={:.2f}({:.2f})\tBatch time={:.2f}({:.2f})\tMemory={:.1f}({:.1f})".format(
                #     global_step, t_total, data_time.val, data_time.avg, batch_time.val, batch_time.avg, memory_meter.val, memory_meter.avg))

        # train_loader = train_loader[0]
        # for step, batch in enumerate(train_loader):
        #     data_time.update(time.time() - end)
        #     batch = tuple(t.to(args.device) for t in batch)

        #     x, y = batch

        #     pred = model(x, 0)
        #     loss_ = loss_fct(pred, y)
            
        #     loss_.backward()
        #     model.allreduce_params()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     MB = 1024 * 1024
        #     memory_meter.update(torch.cuda.max_memory_allocated() / MB)
        #     log.info("Training ({}/{} Steps)\tData time={:.2f}({:.2f})\tBatch time={:.2f}({:.2f})\tMemory={:.1f}({:.1f})".format(
        #         global_step, t_total, data_time.val, data_time.avg, batch_time.val, batch_time.avg, memory_meter.val, memory_meter.avg))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            MB = 1024 * 1024
            memory_meter.update(torch.cuda.max_memory_allocated() / MB)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()
            # optimizer.step()
            # optimizer.zero_grad()
            global_step += 1

            if global_step % 50 == 0:
                loss_string = "".join(["{}: {:.4f}\t".format(k, v.avg) for k, v in losses.items()])
                log.info("Training ({}/{} Steps)\t(loss={})\tData time={:.2f}({:.2f})\tBatch time={:.2f}({:.2f})\tMemory={:.1f}({:.1f})".format(
                    global_step, t_total, loss_string, data_time.val, data_time.avg, batch_time.val, batch_time.avg, memory_meter.val, memory_meter.avg))
            if args.local_rank in [-1, 0]:
                for k, v in losses.items():
                    writer.add_scalar("train/loss_{}".format(k), scalar_value=v.val, global_step=global_step)
                writer.add_scalar("train/total_loss", scalar_value=losses['total'].val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

            if global_step % args.eval_every == 0:
                accuracy_dict = valid(args, model, writer, val_loader, global_step, log, task_list)
                # average accuracy for all tasks
                accuracy = sum(accuracy_dict.values()) / len(accuracy_dict)

            if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                # TODO: evaluate multitask performance, compare and save best model
                if accuracy > best_acc:
                    best_acc = accuracy
                    # save_model(args, model, log)
                    misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=global_step)

                model.train()
                log.info("Current Accuracy: \t{}".format(accuracy))
                log.info('Max accuracy: {:.4f}'.format(best_acc))
            torch.distributed.barrier()

            if global_step % t_total == 0:
                break
        for k, v in loss_dict.items():
            losses[k].reset()
        if global_step % t_total == 0:
            break

    log.info("Final Accuracy: \t{}".format(best_acc))
    log.info("End Training!")

    if args.local_rank in [-1, 0]:
        writer.close()


def get_second_path(path, insert_name="_logs4.17"):
    dir = ""
    root=path
    while dir == "":
        root, dir = os.path.split(root)
    return os.path.join(root, insert_name, dir)

class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.second_path = get_second_path(path)
        self.local_rank = local_rank
        self.log_name = log_name

        if local_rank == 0:
            os.system("mkdir -p {}".format(self.second_path))

    def info(self, msg):
        if self.local_rank in [0, -1]:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")
            # with open(os.path.join(self.second_path, self.log_name), 'a') as f:
            #     f.write(msg + "\n")

def set_seed(args):
    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed + args.local_rank)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "aircraft",
                                              "Pet37", "flowers", "stanford_car",
                                              "cub200", "food101"],
                        default="cifar10", help="Which downstream task.")
    parser.add_argument("--data", default="placeholder", help="Which downstream task.")
    parser.add_argument("--customSplit", default="", help="the downstream custom split.")

    parser.add_argument("--backbone", choices=["deit_small_patch16", "vit_base_patch16", "ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="pretrain/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default=".", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    # bitfit
    parser.add_argument('--bitfit', action="store_true", help="if employing bitfit")

    parser.add_argument('--new_backrazor', action="store_true", help="if employing the backrazor")
    parser.add_argument('--back_prune_ratio', type=float, default=0.8, help="the back prune ratio")
    parser.add_argument('--quantize', action="store_true", help="while pruning, also do the quantization")

    parser.add_argument('--fix_backbone', action="store_true", help="if fix the backbone")

    # profile
    parser.add_argument('--memory_cost_profile', action="store_true", help="profile memory cost")

    # fine-tune
    parser.add_argument('--HeadLr10times', action="store_true", help="increase head lr by 10 times")
    parser.add_argument('--train_resize_first', action="store_true", help="resize the image before "
                                                                          "doing other training augmentations")
    parser.add_argument('--cotuning_trans', action="store_true", help="Employ the trans of cotuning")
    parser.add_argument('--color-distort', action="store_true", help="Employ the color distort")
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # Reprogramming parameters
    parser.add_argument('--main_branch_index', default=None, type=str, help='indices for main branch blocks to be trained, eg, "3,7,11"')
    parser.add_argument('--moe_index', default=None, type=str, help='indices for main branch blocks for moe layers, eg, "3,7,11"')
    parser.add_argument('--reprogram_index', default=None, type=str, help='number of blocks skipped between activation connection')
    parser.add_argument('--fuse_token', action='store_true', help='whether to fuse the inattentive tokens')
    parser.add_argument('--base_keep_rate', type=float, default=0.7,
                        help='Base keep rate (default: 0.7)')
    # parser.add_argument('--shrink_epochs', default=0, type=int, 
    #                     help='how many epochs to perform gradual shrinking of inattentive tokens')
    # parser.add_argument('--shrink_start_epoch', default=5, type=int, 
    #                     help='on which epoch to start shrinking of inattentive tokens')
    parser.add_argument('--drop_loc',  default=None, type=str, help='the layer indices for shrinking inattentive tokens')

    # MoE
    parser.add_argument('--moe_data_distributed', action='store_true', help='if employ moe data distributed')
    parser.add_argument('--moe_gate_type', default="noisy", type=str)
    parser.add_argument('--moe_experts', default=16, type=int, help='moe experts number')
    parser.add_argument('--moe_mlp_ratio', default=None, type=int, help='moe experts mlp ratio')
    parser.add_argument('--moe_top_k', default=None, type=int, help='top k expert number')
    parser.add_argument('--backbone_random_init',default=False, help='whether randomly initialize backbone')
    parser.add_argument('--vmoe_noisy_std', default=1, type=float)
    parser.add_argument('--multi_gate',default=False, help='whether use Multi gate MoE')
    parser.add_argument("--local-rank", type=int, default=-1)


    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        args.distributed = True
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                            timeout=timedelta(minutes=60))
        args.n_gpu = torch.distributed.get_world_size()
        if args.train_batch_size % args.n_gpu != 0:
            raise ValueError("batch size of {} is not divisible by gpu number of {}".format(args.train_batch_size, args.n_gpu))
        args.train_batch_size = args.train_batch_size // args.n_gpu
        args.world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.barrier()

    args.device = device

    # Setup logging
    """ Train the model """
    save_dir = os.path.join(args.output_dir, "checkpoints", args.name)
    writer = None
    if args.local_rank in [-1, 0]:
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs", args.name))

    log = logger(save_dir, log_name="log.txt", local_rank=args.local_rank)
    log.info("Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format
        (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    log.info(str(args))

    # Set seed
    set_seed(args)

    # build task dictionary and dataloaders
    task_dict = {}
    # task_list = ["cifar10"]
    task_list = ["cifar10", "cifar100", "Pet37", "flowers", "food101"]
    train_loader_list, val_loader_list, test_loader_list = [], [], []
    for task in task_list:
        args.dataset = task
        train_loader, val_loader, test_loader, num_classes = get_loader(args)
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
        test_loader_list.append(test_loader)
        task_dict[task] = {'num_classes': num_classes}
    
    # Model & Tokenizer Setup
    # args, model = setup(args, log, num_classes)
    args.drop_loc = None if args.drop_loc is None else [int(x) for x in args.drop_loc.split(',')]
    model = get_model(
            args=args,
            task_dict=task_dict
            # global_pool=args.global_pool
        )
    # Setting gradients false for all layers except the reprogramming layers and classification head
    trainable_blocks = []
    if args.main_branch_index is not None:
        trainable_blocks = args.main_branch_index.split(',')
        preserve_layers = ['reprogram', 'decoders'] + trainable_blocks
        log.info(f'layers preserving gradients:{preserve_layers}')
        # preserve_layers = ['reprogram', 'head', 'cls_token', 'pos_embed', 'patch_embed']
        for name, param in model.named_parameters():
            if not any(x in name for x in preserve_layers):
                log.info('Setting gradients false for layer: {}'.format(name))
                param.requires_grad = False
        # Set last layer norm as trainable
        for name, param in model.named_parameters():
            if 'norm.weight' in name or 'norm.bias' in name:
                log.info('Setting gradients true for layer: {}'.format(name))
                param.requires_grad = True

        # This is to skip memory calculation for the blocks preceding the first trainable block
        # train_blocks = [int(x) for x in trainable_blocks]
        # train_blocks.sort()
        # if not any(x in preserve_layers for x in ['cls_token', 'pos_embed', 'patch_embed']):
        #     no_grad_blocks = [model.backbone.blocks[i] for i in range(12) if i < int(train_blocks[0])]
        #     for block in no_grad_blocks:
        #         for m in block.modules():
        #             if isinstance(m, Attention) or isinstance(m, MainAttention) or isinstance(m, nn.GELU):
        #                 setattr(m, 'requires_backward', False)
        #                 log.info(f'Setting grad for module {m} as False')
        
    # Load pretrained model
    if args.pretrained_dir.endswith('.npz'):
        model.backbone.load_from(np.load(args.pretrained_dir), zero_head=True)
    else:
        checkpoint = torch.load(args.pretrained_dir, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.pretrained_dir)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)


    if not torch.cuda.is_available():
        raise NotImplementedError()
        log.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
            if isinstance(model.backbone, ViTMoE) and (not args.moe_data_distributed):
                print('Use fast moe distributed learning==================>>')
                model = fmoe.DistributedGroupedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True,)
                sync_weights(model, except_key_words=["mlp.experts.h4toh", "mlp.experts.htoh4"])
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
        else:
            model.cuda()
            if isinstance(model.backbone, ViTMoE) and (not args.moe_data_distributed):
                model = fmoe.DistributedGroupedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.local_rank is not None:
        model.cuda()
    else:
        raise NotImplementedError()


    model.to(args.device)
    if args.fix_backbone:
        # assert not ("resnet" in args.model_type)
        for name, parameter in model.named_parameters():
            if "decoder" in name:
                log.info("fix_backbone, not freeze {}".format(name))
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    # if args.new_backrazor and args.quantize:
    #     for name, module in model.named_modules():
    #         module.name = name
    #     ms.policy.deploy_on_init(model, 'ViT/policy_tiny-8bit.txt', verbose=print, override_verbose=False)
    #     model.to(args.device)

    log.info(str(model))

    activation_bits = 32
    # memory_cost, memory_cost_dict = profile_memory_cost(model, input_size=(1, 3, 224, 224), require_backward=True,
    #                                                     activation_bits=activation_bits, trainable_param_bits=32,
    #                                                     head_only=args.fix_backbone,
    #                                                     frozen_param_bits=8, batch_size=128)
    # MB = 1024 * 1024
    # log.info("memory_cost is {:.1f} MB, param size is {:.1f} MB, act_size each sample is {:.1f} MB".
    #          format(memory_cost / MB, memory_cost_dict["param_size"] / MB, memory_cost_dict["act_size"] / MB))

    log.info("Dataset {}, Train dataset len is {}, test dataset len is {}".
             format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))
    
    # Profile MACs
    from thop import profile
    from thop import clever_format
    sample_input = torch.randn((128, 3, 224, 224)).to(device)
    model.eval()
    # for ii, sample in enumerate(val_loader_list[0]):
        # sample = val_dataloader[0]
        # inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
        # assert inputs.size(0)==1
    flops, params = profile(model, inputs=(sample_input, 0),)
    flops, params = clever_format([flops, params], "%.3f")
    # if p['backbone']=='VisionTransformer':
    #     flops = int(flops)+int(model.backbone.flops())
    print(flops,params)
    # exit()
    
    # Training
    # Prepare dataset
    train(args, model, train_loader_list, val_loader_list, task_list, log, writer)


if __name__ == "__main__":
    main()
