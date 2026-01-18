cd .//MambaGalerkinTransformer

task_name="st_darcyflow_pdebench"
rm -r log_train_${task_name}.txt

export CUDA_VISIBLE_DEVICES=0
nohup \
python train_darcyflow_pdebench.py \
--config-path config/darcyflow_pdebench/config_${task_name}.yml \
>> log_train_${task_name}.txt &