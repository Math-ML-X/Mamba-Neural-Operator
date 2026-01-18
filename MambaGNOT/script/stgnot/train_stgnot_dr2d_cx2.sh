cd .//MambaGNOT

task_name="stgnot_dr2d_cx2"
rm -r log_train_${task_name}.txt

nohup \
python train_dr2d_pdebench.py \
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
--val-batch-size 4 \
--normalize_x none \
--use-normalizer unit \
--seed 42 \
>> log_train_${task_name}.txt &
