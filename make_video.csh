SIGMA=25
SEQ=camel
RESULTS_DIR=results_kvbank_residual_loss_10

ffmpeg \
  -framerate 5 \
  -pattern_type glob -i "${RESULTS_DIR}/${SEQ}/groundtruth/gt_sigma${SIGMA}_*.png" \
  -framerate 5 \
  -pattern_type glob -i "${RESULTS_DIR}/${SEQ}/noisy/noisy_sigma${SIGMA}_*.png" \
  -framerate 5 \
  -pattern_type glob -i "${RESULTS_DIR}/${SEQ}/denoised/denoised_sigma${SIGMA}_*.png" \
  -filter_complex "
    [0:v]drawtext=text='Clean':fontsize=28:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5[v0];
    [1:v]drawtext=text='Noisy sigma=${SIGMA}':fontsize=28:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5[v1];
    [2:v]drawtext=text='Denoised (${RESULTS_DIR})':fontsize=28:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5[v2];
    [v0][v1][v2]hstack=inputs=3[out]
  " \
  -map "[out]" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p \
  ${SEQ}_sigma${SIGMA}_${RESULTS_DIR}.mp4