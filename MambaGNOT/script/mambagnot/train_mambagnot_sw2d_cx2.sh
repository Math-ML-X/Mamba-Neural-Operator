cd .//MambaGNOT

task_name="mambagnot_sw2d_cx2"
rm -r log_train_${task_name}.txt

nohup \
python train_sw2d_pdebench.py \
--model-name MambaGNOT \
--gpu 3 \
--dataset sw2d_cx2 \
--comment mambagnot_sw2d_cx2_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 8 \
--val-batch-size 8 \
--normalize_x none \
--use-normalizer unit \
--seed 42 \
>> log_train_${task_name}.txt &
