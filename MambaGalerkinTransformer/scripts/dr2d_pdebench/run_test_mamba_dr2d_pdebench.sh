cd .//MambaGalerkinTransformer

task_name="mamba_dr2d_pdebench"
rm -r log_test_${task_name}.txt

export CUDA_VISIBLE_DEVICES=1
nohup \
python test_dr2d_pdebench.py \
--config-path config/dr2d_pdebench/config_${task_name}.yml \
--weight-path ./Mamba_Transformer_PDE/MambaGalerkinTransformer/runs/dr2d_pdebench/dr2d_pdebench_mb_Cx1_20240615141845/dr2d_pdebench_mb_Cx1.pt \
--is_vis \
>> log_test_${task_name}.txt &