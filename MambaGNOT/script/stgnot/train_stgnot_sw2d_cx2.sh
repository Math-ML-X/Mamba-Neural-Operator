cd .//MambaGNOT

task_name="stgnot_sw2d_cx2"
rm -r log_train_${task_name}.txt

nohup \
python train_sw2d_pdebench.py \
--model-name STGNOT \
--attn-type standard \
--gpu 1 \
--dataset sw2d_cx2 \
--comment stgnot_sw2d_cx2_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 4 \
--val-batch-size 4 \
--normalize_x none \
--use-normalizer unit \
>> log_train_${task_name}.txt &
