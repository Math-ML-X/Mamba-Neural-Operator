cd .//MambaGNOT

task_name="gnot_dr2d"
rm -r log_test_${task_name}.txt

nohup \
python test_dr2d_pdebench.py \
--weight-path ./Mamba_Transformer_PDE/MambaGNOT/runs/dr2d_pdebench/gnot_dr2d_mse_20240619112842/checkpoints/dr2d_all_MIOEGPTgnot_dr2d_mse_0619_11_28_42.pt \
--model-name GNOT \
--gpu 0 \
--dataset dr2d \
--comment gnot_dr2d_mse \
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

