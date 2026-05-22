python test_fastdvdnet.py \
  --model_file logs_actual_noise/net_best.pth \
  --test_path data/val/ \
  --noise_sigma 10 \
  --lam 10 \
  --max_num_fr_per_seq 100 \
  --save_path results_kvbank_actual_noise 