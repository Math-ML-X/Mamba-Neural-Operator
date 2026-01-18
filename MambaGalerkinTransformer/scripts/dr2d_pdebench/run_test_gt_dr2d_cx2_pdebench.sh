cd .//MambaGalerkinTransformer

task_name="gt_dr2d_cx2_pdebench"
rm -r log_test_${task_name}.txt

export CUDA_VISIBLE_DEVICES=1
nohup \
python test_dr2d_pdebench.py \
--config-path config/dr2d_pdebench/config_${task_name}.yml \
--weight-path ./Mamba_Transformer_PDE/MambaGalerkinTransformer/runs/dr2d_pdebench/dr2d_pdebench_gt_Cx2_20240704203315/dr2d_pdebench_gt_Cx2.pt \
>> log_test_${task_name}.txt &