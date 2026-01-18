cd .//MambaGNOT

task_name="stgnot_darcyflow2d"
rm -r log_test_${task_name}.txt

nohup \
python test_darcyflow2d_pdebench.py \
--weight-path ./Mamba_Transformer_PDE/MambaGNOT/runs/darcyflow_pdebench/stgnot_darcyflow2d_mse_20240627103333/checkpoints/darcyflow2d_all_STMIOEGPTstgnot_darcyflow2d_mse_0627_10_33_34.pt \
--model-name STGNOT \
--attn-type standard \
--gpu 1 \
--dataset darcyflow2d \
--comment stgnot_darcyflow2d_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 6 \
--val-batch-size 6 \
--normalize_x none \
--use-normalizer none \
--is_vis \
>> log_test_${task_name}.txt &


