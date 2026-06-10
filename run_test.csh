python test_fastdvdnet.py \
  --model_file logs_test_v3/net_best.pth \
  --test_path data_fullhd/val/ \
  --noise_sigma 10 \
  --lam 10 \
  --max_num_fr_per_seq 100 \
  --save_path results_test 