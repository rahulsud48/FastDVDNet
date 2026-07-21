FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PIP_PROGRESS_BAR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV GOTO_NUM_THREADS=1

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        opencv-python-headless \
        numpy \
        scikit-image \
        tensorboardX \
        tqdm \
        onnx==1.17.0 \
        onnxruntime==1.20.1

WORKDIR /workspace/fastdvdnet

CMD ["/bin/bash"]
