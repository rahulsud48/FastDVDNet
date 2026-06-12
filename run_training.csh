python train_fastdvdnet.py \
  --trainset_dir data/train \
  --valset_dir data/val \
  --log_dir logs_kvbank_yuv420_32_32_64 \
  --batch_size 4 \
  --epochs 100 \
  --max_number_patches 256000 \
  --yuv