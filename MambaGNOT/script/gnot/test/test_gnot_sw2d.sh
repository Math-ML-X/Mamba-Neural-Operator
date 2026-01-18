cd .//MambaGNOT

task_name="gnot_sw2d"
rm -r log_test_${task_name}.txt

nohup \
python test_sw2d_pdebench.py \
--weight-path ./Mamba_Transformer_PDE/MambaGNOT/runs/sw2d_pdebench/gnot_sw2d_mse_20240619112843/checkpoints/sw2d_all_MIOEGPTgnot_sw2d_mse_0619_11_28_44.pt \
--model-name GNOT \
--gpu 0 \
--dataset sw2d \
--comment gnot_sw2d_mse \
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
