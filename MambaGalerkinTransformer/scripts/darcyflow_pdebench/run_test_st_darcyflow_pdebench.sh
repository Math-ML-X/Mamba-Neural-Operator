cd .//MambaGalerkinTransformer

task_name="st_darcyflow_pdebench"
rm -r log_test_${task_name}.txt

export CUDA_VISIBLE_DEVICES=3
nohup \
python test_darcyflow_pdebench.py \
--config-path config/darcyflow_pdebench/config_${task_name}.yml \
--weight-path ./Mamba_Transformer_PDE/MambaGalerkinTransformer/runs/darcyflow_pdebench/darcyflow_pdebench_st_Stanx2_20240627103506/darcyflow_pdebench_st_Stanx2.pt \
--is_vis \
>> log_test_${task_name}.txt &