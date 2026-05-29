python test_fastdvdnet.py \
    --model_file logs_y_only_v2/net_best.pth \
    --test_path data/val \
    --save_path result_y_only_v2 \
    --noise_sigma 50 \
    --save_y_only \
    --max_num_fr_per_seq 100