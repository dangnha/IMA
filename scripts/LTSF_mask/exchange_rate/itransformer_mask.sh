export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 64 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001