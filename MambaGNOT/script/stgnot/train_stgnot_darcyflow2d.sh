cd .//MambaGNOT

task_name="stgnot_darcyflow2d"
rm -r log_train_${task_name}.txt

nohup \
python train_darcyflow2d_pdebench.py \
--model-name STGNOT \
--attn-type standard \
--gpu 1 \
--dataset darcyflow2d \
--comment stgnot_darcyflow2d_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 6 \
--val-batch-size 6 \
--normalize_x none \
--use-normalizer none \
>> log_train_${task_name}.txt &
