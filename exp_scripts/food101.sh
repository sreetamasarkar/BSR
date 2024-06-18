save_dir="."

devices="4"
port=6732
n_gpu=1

# backPruneRatio=0.8

lr=0.01

# Baseline
# CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
# train.py --name food101-lr${lr}-B128 --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
# --dataset food101 --model_type vit_base_patch16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
# --train_batch_size 128 --eval_batch_size 128 \
# --num_steps 30000 --eval_every 1000
# # --new_backrazor --back_prune_ratio ${backPruneRatio} \

# Head Only
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name food101-lr${lr}-B128-fix-backbone --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset food101 --model_type vit_base_patch16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 30000 --eval_every 1000 --fix_backbone

# Reprogramming
# main_branch_index="3,7,11"
# drop_loc="3,6,9"
# base_keep_rate=0.5

# CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
# train.py --name food101-lr${lr}-B128-train${main_branch_index}-drop${drop_loc}-kr${base_keep_rate} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
# --dataset food101 --model_type vit_base_patch16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
# --train_batch_size 128 --eval_batch_size 128 \
# --num_steps 30000 --eval_every 100 \
# --main_branch_index ${main_branch_index} --drop_loc ${drop_loc} --base_keep_rate ${base_keep_rate} \
# --fuse_token \

