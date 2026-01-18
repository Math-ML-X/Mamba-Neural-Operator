cd .//MambaOFormer

export CUDA_VISIBLE_DEVICES=0
task_name=OFormer_SW2D_Cx1_Tx1
rm -r log_test_${task_name}.txt
nohup python train_shallow_water_mamba.py \
--ckpt_every 5000 \
--iters 100000 \
--lr 1e-4 \
--batch_size 4 \
--in_seq_len 10 \
--out_seq_len 91 \
--dataset_path ./data/PDE/data/ShallowWater/PDEBench/npz/train \
--in_channels 12 \
--out_channels 1 \
--encoder_emb_dim 64 \
--out_seq_emb_dim 64 \
--encoder_depth 2 \
--decoder_emb_dim 128 \
--propagator_depth 1 \
--out_step 1 \
--train_seq_num 900 \
--test_seq_num 100 \
--fourier_frequency 8 \
--encoder_heads 1 \
--use_grad \
--curriculum_ratio 0.0 \
--curriculum_steps 0 \
--aug_ratio 0.0 \
--reduce_resolution 1 \
--reduce_resolution_t 1 \
--task_name ${task_name} \
--eval_mode \
--is_vis \
--path_to_resume ./Mamba_Transformer_PDE/MambaOFormer/runs/sw2d_pdebench/OFormer_SW2D_Cx1_Tx1_BK/sw2d_latest_model.ckpt \
>> log_test_${task_name}.txt &

