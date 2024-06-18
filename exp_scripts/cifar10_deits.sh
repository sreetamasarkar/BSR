save_dir="."

devices="2"
port=7238
n_gpu=1

# model_type="vit_base_patch16"
# pretrained_model_path="${save_dir}/pretrain/ViT-B_16.npz"
# model_type="deit_small_patch16"
# pretrained_model_path="${save_dir}/deit_small_patch16_224-cd65a155.pth"

# Baseline
# python main_finetune.py --finetune deit_small_patch16_224-cd65a155.pth --model deit_small_patch16 --dataset cifar10 --nb_classes 10 \
# --batch_size 128 --output_dir output_dir/cifar10/deits_ep50_b128 --log_dir output_dir/cifar10/deits_ep50_b128 --device cuda:${devices}


# Reprogramming
main_branch_index="3,7,11"
drop_loc="3,6,9"
base_keep_rate=0.5

python main_reprogram.py --dataset cifar10 --nb_classes=10 --model deit_small_patch16 --finetune pretrain/deit_small_patch16_224-cd65a155.pth \
--batch_size=128 --output_dir output_dir/cifar10/deits_train_${main_branch_index}_drop${drop_loc}_kr${base_keep_rate} --log_dir output_dir/cifar10/deits_train_${main_branch_index}_drop${drop_loc}_kr${base_keep_rate} \
--main_branch_index ${main_branch_index} --drop_loc ${drop_loc} --base_keep_rate ${base_keep_rate} \
--fuse_token --device cuda:${devices}