export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet
  python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_mask_0.375 \
  --mask_rate 0.375 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001