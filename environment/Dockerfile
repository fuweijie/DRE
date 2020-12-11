# hash:sha256:ae607b92d68b02da597575fceac908224d0f25a4f5d620eae869c9c4bdaf5e0a
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        matplotlib==3.1.3 \
        more-itertools==8.3.0 \
        numpy==1.18.1 \
        pillow==7.1.2 \
        pytorch==1.1.0 \
        scikit-image==0.16.2 \
        scikit-learn==0.22.1 \
        scipy==1.4.1 \
        torchvision==0.3.0 \
    && conda clean -ya
