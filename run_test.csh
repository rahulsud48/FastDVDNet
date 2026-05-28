python test_fastdvdnet.py \
  --model_file logs_y_only_v2/net_best.pth \
  --test_path data/val/ \
  --noise_sigma 25 \
  --save_path result_y_only_v2 \
  --max_num_fr_per_seq 100