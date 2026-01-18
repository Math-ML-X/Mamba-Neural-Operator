cd .//MambaGNOT

task_name="mambagnot_darcyflow2d"
rm -r log_test_${task_name}.txt

nohup \
python test_darcyflow2d_pdebench.py \
--weight-path ./Mamba_Transformer_PDE/MambaGNOT/runs/darcyflow_pdebench/mambagnot_darcyflow2d_mse_20240619110713/checkpoints/darcyflow2d_all_MambaMIOEGPTmambagnot_darcyflow2d_mse_0619_11_07_14.pt \
--model-name MambaGNOT \
--gpu 1 \
--dataset darcyflow2d \
--comment mambagnot_darcyflow2d_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 8 \
--val-batch-size 8 \
--normalize_x none \
--use-normalizer none \
--is_vis \
>> log_test_${task_name}.txt &

