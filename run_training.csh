python train_fastdvdnet.py \
  --trainset_dir data/train \
  --valset_dir data/val \
  --log_dir logs_test_qat_smoke \
  --batch_size 4 \
  --epochs 100 \
  --max_number_patches 8 \
  --yuv \
  --qat