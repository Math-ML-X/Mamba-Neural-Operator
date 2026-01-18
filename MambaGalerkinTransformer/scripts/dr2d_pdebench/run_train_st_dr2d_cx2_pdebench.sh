cd .//MambaGalerkinTransformer

task_name="st_dr2d_cx2_pdebench"
rm -r log_train_${task_name}.txt

export CUDA_VISIBLE_DEVICES=1
nohup \
python train_dr2d_pdebench.py \
--config-path config/dr2d_pdebench/config_${task_name}.yml \
>> log_train_${task_name}.txt &