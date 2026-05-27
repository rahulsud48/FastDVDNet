python train_fastdvdnet.py \
  --trainset_dir data/train \
  --valset_dir data/val \
  --log_dir logs_actual_noise \
  --batch_size 4 \
  --epochs 80 \
  --max_number_patches 256000 \
  --resume_training