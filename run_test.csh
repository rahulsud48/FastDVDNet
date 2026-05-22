python test_fastdvdnet.py \
  --model_file logs_kv_bank_shot_read_flicker/net_best.pth \
  --test_path data/val \
  --save_path results_low_light \
  --config configs/test_low_light.json \
  --max_num_fr_per_seq 100