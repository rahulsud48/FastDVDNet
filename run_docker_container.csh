docker run --gpus all -it --rm \
  --security-opt seccomp=unconfined \
  --user $(id -u):$(id -g) \
  -e OPENBLAS_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  -e VECLIB_MAXIMUM_THREADS=1 \
  -e GOTO_NUM_THREADS=1 \
  -v /media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a391/MMAlgo/TNR/fastdvdnet:/workspace/fastdvdnet \
  -w /workspace/fastdvdnet \
  fastdvdnet:latest