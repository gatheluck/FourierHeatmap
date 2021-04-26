FROM nvidia/cuda:11.0-devel-ubuntu20.04
ARG PYTHON_VERSION=3.9.2
ARG SCRATCH_DIRECTORY=/root/scratch

RUN apt update && apt install --no-install-recommends -y \
	git curl ssh openssh-client \
	python3.9 python3-pip \
	&& pip3 install poetry

WORKDIR ${SCRATCH_DIRECTORY}
RUN git clone https://github.com/gatheluck/FourierHeatmap.git

WORKDIR ${SCRATCH_DIRECTORY}/FourierHeatmap
RUN poetry install

CMD poetry run python fhmap/apps/eval_fhmap.py \
	arch=${ARCH} \
	batch_size=${BATCH_SIZE} \
	dataset=${DATASET} \
	eps=${EPS} \
	ignore_edge_size=${IGNORE_EDGE_SIZE} \
	num_samples=${NUM_SAMPLES} \
	weightpath=/root/scratch/FourierHeatmap/weight/${WEIGHTFILE}
