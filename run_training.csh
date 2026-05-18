python train_fastdvdnet.py \
  --trainset_dir data/train \
  --valset_dir data/val \
  --log_dir logs_kv_bank_noise_model \
  --batch_size 4 \
  --epochs 80 \
  --temporal_mode random \
  --spatial_mode random \
  --max_number_patches 256000 