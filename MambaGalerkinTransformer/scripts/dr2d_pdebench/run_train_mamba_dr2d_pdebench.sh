cd .//MambaGalerkinTransformer

task_name="mamba_dr2d_pdebench"
rm -r log_train_${task_name}.txt

export CUDA_VISIBLE_DEVICES=3
nohup \
python train_dr2d_pdebench.py \
--config-path config/dr2d_pdebench/config_${task_name}.yml \
>> log_train_${task_name}.txt &