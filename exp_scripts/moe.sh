# MoE
# Baseline
# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 --master_port 3050 train_moe.py --name moe_3711_baseline --backbone vit_base_patch16 --moe_gate_type noisy_vmoe --moe_experts 16 --moe_top_k 4 --backbone_random_init False --vmoe_noisy_std 0 --multi_gate True --moe_mlp_ratio 4 --train_batch_size=128 --learning_rate=0.01 --moe_index="3,7,11"

# Ours
# Reprogramming
moe_index="3,7,11"
main_branch_index="3,7,11"
drop_loc="3,6,9"
base_keep_rate=0.5

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 --master_port 3350 train_moe.py --name moe_${moe_index}_drop_${drop_loc}_train_${main_branch_index} --backbone vit_base_patch16 --moe_gate_type noisy_vmoe --moe_experts 16 --moe_top_k 4 --backbone_random_init False --vmoe_noisy_std 0 --multi_gate True --moe_mlp_ratio 4 --train_batch_size=128 --learning_rate=0.01 --main_branch_index ${main_branch_index} --moe_index ${moe_index} --drop_loc ${drop_loc}