# train SMD model
# change "model_name" parameter to choose different method
python -u run.py \
  --dataset_root_path ./data/SMD/ \
  --dataset_filename SMD.csv \
  --model_des SMD_train \
  --description SMD_train \
  --dataset_field SMD \
  --model_name ns_Autoformer \
  --forcast_task S \
  --seq_len 464 \
  --label_len 48 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --gpu 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --itr 1 &

# test only
python -u run.py \
  --no_training \
  --dataset_root_path ./data/SMD/ \
  --dataset_filename SMD.csv \
  --model_des SMD_train \
  --description SMD_test \
  --model_name ns_Autoformer \
  --dataset_field SMD \
  --forcast_task S \
  --seq_len 464 \
  --label_len 48 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --gpu 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --itr 1 &