cd .//MambaGNOT

task_name="gnot_darcyflow2d"
rm -r log_test_${task_name}.txt

nohup \
python test_darcyflow2d_pdebench.py \
--weight-path ./Mamba_Transformer_PDE/MambaGNOT/runs/darcyflow_pdebench/gnot_darcyflow2d_mse_20240619110930/checkpoints/darcyflow2d_all_MIOEGPTgnot_darcyflow2d_mse_0619_11_09_31.pt \
--model-name GNOT \
--gpu 1 \
--dataset darcyflow2d \
--comment gnot_darcyflow2d_mse \
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

