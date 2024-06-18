# MTL

# Baseline
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 3250 train_moe.py 
--name mtl_baseline --backbone vit_base_patch16 --backbone_random_init False --train_batch_size=128 --learning_rate=0.01 
