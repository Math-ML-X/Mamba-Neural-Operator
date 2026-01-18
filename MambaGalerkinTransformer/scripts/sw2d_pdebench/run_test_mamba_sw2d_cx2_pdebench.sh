cd .//MambaGalerkinTransformer

task_name="mamba_sw2d_cx2_pdebench"
rm -r log_test_${task_name}.txt

export CUDA_VISIBLE_DEVICES=2
nohup \
python test_sw2d_pdebench.py \
--config-path config/sw2d_pdebench/config_${task_name}.yml \
--weight-path ./Mamba_Transformer_PDE/MambaGalerkinTransformer/runs/sw2d_pdebench/sw2d_pdebench_mb_Cx2_20240704203437/sw2d_pdebench_mb_Cx2.pt \
>> log_test_${task_name}.txt &