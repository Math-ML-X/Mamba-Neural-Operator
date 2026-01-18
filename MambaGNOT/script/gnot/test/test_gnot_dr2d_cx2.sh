cd .//MambaGNOT

task_name="gnot_dr2d_cx2"
rm -r log_test_${task_name}.txt

nohup \
python test_dr2d_pdebench.py \
--weight-path ./Mamba_Transformer_PDE/MambaGNOT/runs/dr2d_pdebench/gnot_dr2d_cx2_mse_20240704202014/checkpoints/dr2d_stanx2_all_MIOEGPTgnot_dr2d_stanx2_mse_0704_20_20_15.pt \
--model-name GNOT \
--gpu 0 \
--dataset dr2d_cx2 \
--comment gnot_dr2d_cx2_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 8 \
--val-batch-size 8 \
--normalize_x none \
--use-normalizer unit \
>> log_test_${task_name}.txt &

