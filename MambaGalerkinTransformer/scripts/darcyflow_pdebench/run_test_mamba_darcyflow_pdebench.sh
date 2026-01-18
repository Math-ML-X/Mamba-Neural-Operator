cd .//MambaGalerkinTransformer

task_name="mamba_darcyflow_pdebench"
rm -r log_test_${task_name}.txt

export CUDA_VISIBLE_DEVICES=3
nohup \
python test_darcyflow_pdebench.py \
--config-path config/darcyflow_pdebench/config_${task_name}.yml \
--weight-path ./Mamba_Transformer_PDE/MambaGalerkinTransformer/runs/darcyflow_pdebench/darcyflow_pdebench_mb_Stanx2_20240616171259/darcyflow_pdebench_mb_Stanx2.pt \
--is_vis \
>> log_test_${task_name}.txt &