FROM nvidia/cuda:11.0-devel-ubuntu20.04
ARG SCRATCH_DIRECTORY=/root/scratch

WORKDIR ${SCRATCH_DIRECTORY}
RUN apt update && apt install --no-install-recommends -y \
	git curl ssh openssh-client\
	python3 python3-pip \
	&& pip3 install poetry

RUN git clone https://github.com/gatheluck/FourierHeatmap.git

WORKDIR ${SCRATCH_DIRECTORY}/FourierHeatmap
RUN poetry install
RUN ln -s /var/data data
RUN mkdir outputs && ln -s /var/outputs outputs