python test_fastdvdnet.py \
  --model_file logs_qat_check/net_best.pth \
  --test_path data_fullhd/val/ \
  --noise_sigma 25 \
  --lam 25 \
  --max_num_fr_per_seq 100 \
  --save_path results_qat_check \
  --yuv \
  --qat