cd .//MambaGNOT

task_name="mambagnot_sw2d"
rm -r log_train_${task_name}.txt

nohup \
python train_sw2d_pdebench.py \
--model-name MambaGNOT \
--gpu 2 \
--dataset sw2d \
--comment mambagnot_sw2d_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 8 \
--val-batch-size 8 \
--normalize_x none \
--use-normalizer unit \
>> log_train_${task_name}.txt &
