python test_fastdvdnet.py \
  --model_file logs_kv_bank_dwconv_strided_with_gate_skips/net_best.pth \
  --test_path data/val/ \
  --noise_sigma 25 \
  --lam 25 \
  --max_num_fr_per_seq 100 \
  --save_path kv_bank_dwconv_strided_with_gate_skips 