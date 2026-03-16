python train_fastdvdnet.py \
  --trainset_dir data_overfit/train \
  --valset_dir data_overfit/val \
  --log_dir logs_overfit \
  --batch_size 2 \
  --epochs 500 \
  --max_number_patches 256000 \
  --no_orthog