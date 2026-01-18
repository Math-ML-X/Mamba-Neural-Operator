cd .//MambaGNOT

task_name="stgnot_dr2d_cx2"
rm -r log_test_${task_name}.txt

nohup \
python test_dr2d_pdebench.py \
--weight-path ./Mamba_Transformer_PDE/MambaGNOT/runs/dr2d_pdebench/stgnot_dr2d_cx2_mse_20240704201456/checkpoints/dr2d_stanx2_all_STMIOEGPTstgnot_dr2d_stanx2_mse_0704_20_14_57.pt \
--model-name STGNOT \
--attn-type standard \
--gpu 0 \
--dataset dr2d_cx2 \
--comment stgnot_dr2d_cx2_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 4 \
--val-batch-size 2 \
--normalize_x none \
--use-normalizer unit \
>> log_test_${task_name}.txt &


