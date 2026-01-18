cd .//MambaGNOT

task_name="mambagnot_dr2d_cx2"
rm -r log_test_${task_name}.txt

nohup \
python test_dr2d_pdebench.py \
--weight-path ./Mamba_Transformer_PDE/MambaGNOT/runs/dr2d_pdebench/mambagnot_dr2d_cx2_mse_20240704201752/checkpoints/dr2d_stanx2_all_MambaMIOEGPTmambagnot_dr2d_stanx2_mse_0704_20_17_54.pt \
--model-name MambaGNOT \
--gpu 0 \
--dataset dr2d_cx2 \
--comment mambagnot_dr2d_cx2_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 8 \
--val-batch-size 8 \
--normalize_x none \
--use-normalizer unit \
>> log_test_${task_name}.txt &


