python test_fastdvdnet.py \
  --model_file logs_test_qat/net_best.pth \
  --test_path data_fullhd/val/ \
  --noise_sigma 25 \
  --lam 25 \
  --max_num_fr_per_seq 100 \
  --save_path result_test_qat \
  --yuv \
  --qat_int8