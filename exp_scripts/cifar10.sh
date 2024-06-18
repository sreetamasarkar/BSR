save_dir="."

devices="2"
port=7238
n_gpu=1

# devices="0,1"
# port=7256
# n_gpu=2

# backPruneRatio=0.9
lr=0.001
model_type="vit_base_patch16"
pretrained_model_path="${save_dir}/pretrain/ViT-B_16.npz"
# model_type="deit_small_patch16"
# pretrained_model_path="${save_dir}/pretrain/deit_small_patch16_224-cd65a155.pth"

# Baseline
# CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
# train.py --name cifar10-${model_type}-lr${lr}-B128 --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
# --dataset cifar10 --model_type ${model_type} --pretrained_dir ${pretrained_model_path} \
# --train_batch_size 128 --eval_batch_size 128 \
# --num_steps 20000 --eval_every 1000 \


# Reprogramming
main_branch_index="3,7,11"
drop_loc="3,6,9"
base_keep_rate=0.5

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-model${model_type}-B128-train${main_branch_index}-drop${drop_loc}-kr${base_keep_rate} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar10 --model_type ${model_type} --pretrained_dir ${pretrained_model_path} \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 20000 --eval_every 1000 \
--main_branch_index ${main_branch_index} --drop_loc ${drop_loc} --base_keep_rate ${base_keep_rate} \
--fuse_token \



# --reprogram_index ${reprog_index} 

# CUDA_VISIBLE_DEVICES=${devices} python train.py --name cifar10-lr${lr}-B128--reprog${reprog_index} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
# --dataset cifar10 --model_type vit_base_patch16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
# --train_batch_size 128 --eval_batch_size 128 \
# --num_steps 20000 --eval_every 1000 \
# --reprogram_index 3 7 11