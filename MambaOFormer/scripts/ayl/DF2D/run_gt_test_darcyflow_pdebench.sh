cd .//MambaOFormer

export CUDA_VISIBLE_DEVICES=0
task_name=OFormer_DarcyFlow_beta1.0
rm -r log_test_${task_name}.txt
nohup python train_darcyflow_pdebench_mamba.py \
--ckpt_every 5000 \
--iters 200001 \
--lr 1e-4 \
--batch_size 8 \
--train_dataset_path /media/ssd/data_temp/PDE/data/DarcyFlow/PDEBench/2D_DarcyFlow_beta1.0_Train.hdf5 \
--test_dataset_path /media/ssd/data_temp/PDE/data/DarcyFlow/PDEBench/2D_DarcyFlow_beta1.0_Train.hdf5 \
--train_seq_num 9000 \
--test_seq_num 1000 \
--resolution 64 \
--reduce_resolution 2 \
--resolution_query 64 \
--reduce_resolution_query 2 \
--task_name ${task_name} \
--eval_mode \
--is_vis \
--path_to_resume ./Mamba_Transformer_PDE/MambaOFormer/runs/darcyflow_pdebench/OFormer_DarcyFlow_beta1.0_BK/darcy_beta1.0_latest_model.ckpt \
>> log_test_${task_name}.txt &

