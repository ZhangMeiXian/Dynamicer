python -u executor.py \
  --model_des CSM_train \
  --exp_des CSM_train \
  --dataset CSM \
  --region China \
  --model adj_ns_FEDformer \
  --forcast_task M \
  --neighbor_window 20 \
  --n_index 1 \
  --sample_time_window_before 30 \
  --sample_time_window_after 0 \
  --sample_day_window 14 \
  --train_epochs 10 \
  --train_ratio 0.7 \
  --test_ratio 0.1 \
  --anomaly_class_weight 5 \
  --seq_len 464 \
  --label_len 48 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --gpu 0 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --itr 1 