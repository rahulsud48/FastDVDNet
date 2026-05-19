ffmpeg \
  -framerate 30 -i data/val/blackswan/%05d.jpg \
  -framerate 30 -i data/val_noisy/blackswan/%05d.png \
  -filter_complex "[0:v][1:v]hstack=inputs=2" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p \
  blackswan_comparison.mp4