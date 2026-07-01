python train_fastdvdnet.py \
  --trainset_dir data/train \
  --valset_dir data/val \
  --log_dir logs_qat_check \
  --batch_size 4 \
  --epochs 10 \
  --max_number_patches 256000 \
  --yuv \
  --qat