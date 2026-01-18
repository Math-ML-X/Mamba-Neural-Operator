cd .//MambaGNOT

task_name="gnot_darcyflow2d"
rm -r log_train_${task_name}.txt

nohup \
python train_darcyflow2d_pdebench.py \
--model-name GNOT \
--gpu 0 \
--dataset darcyflow2d \
--comment gnot_darcyflow2d_mse \
--epochs 100 \
--loss-name mse \
--train-num all \
--test-num all \
--batch-size 8 \
--val-batch-size 8 \
--normalize_x none \
--use-normalizer none \
>> log_train_${task_name}.txt &
