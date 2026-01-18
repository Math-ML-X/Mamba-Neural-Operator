cd .//MambaGalerkinTransformer

task_name="gt_sw2d_pdebench"
rm -r log_test_${task_name}.txt

export CUDA_VISIBLE_DEVICES=2
nohup \
python test_sw2d_pdebench.py \
--config-path config/sw2d_pdebench/config_${task_name}.yml \
--weight-path ./Mamba_Transformer_PDE/MambaGalerkinTransformer/runs/sw2d_pdebench/sw2d_pdebench_gt_Cx1_20240615141207/sw2d_pdebench_gt_Cx1.pt \
--is_vis \
>> log_test_${task_name}.txt &