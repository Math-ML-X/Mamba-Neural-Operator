cd .//MambaGalerkinTransformer

task_name="gt_sw2d_cx2_pdebench"
rm -r log_train_${task_name}.txt

export CUDA_VISIBLE_DEVICES=0
nohup \
python train_sw2d_pdebench.py \
--config-path config/sw2d_pdebench/config_${task_name}.yml \
>> log_train_${task_name}.txt &