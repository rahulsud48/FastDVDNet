python test_fastdvdnet.py \
  --model_file logs_kv_bank_residual_loss/net_best.pth \
  --test_path ./data/val \
  --save_path ./results_kvbank_residual_loss \
  --noise_sigma 25 \
  --seed 42 \
  --max_num_fr_per_seq 100