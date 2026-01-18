cd .//MambaGNOT

task_name="gnot_sw2d_cx2"
rm -r log_test_${task_name}.txt

nohup \
python test_sw2d_pdebench.py \
--weight-path ./Mamba_Transformer_PDE/MambaGNOT/runs/sw2d_pdebench/gnot_sw2d_cx2_mse_20240704202010/checkpoints/sw2d_stanx2_all_MIOEGPTgnot_sw2d_stanx2_mse_0704_20_20_11.pt \
--model-name GNOT \
--gpu 0 \
--dataset sw2d_cx2 \
--comment gnot_sw2d_cx2_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 8 \
--val-batch-size 8 \
--normalize_x none \
--use-normalizer unit \
>> log_test_${task_name}.txt &

