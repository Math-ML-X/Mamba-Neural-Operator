cd .//MambaGNOT

task_name="mambagnot_dr2d"
rm -r log_test_${task_name}.txt

nohup \
python test_dr2d_pdebench.py \
--weight-path ./Mamba_Transformer_PDE/MambaGNOT/runs/dr2d_pdebench/mambagnot_dr2d_mse_20240619110716/checkpoints/dr2d_all_MambaMIOEGPTmambagnot_dr2d_mse_0619_11_07_17.pt \
--model-name MambaGNOT \
--gpu 0 \
--dataset dr2d \
--comment mambagnot_dr2d_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 8 \
--val-batch-size 8 \
--normalize_x none \
--use-normalizer unit \
--is_vis \
>> log_test_${task_name}.txt &

