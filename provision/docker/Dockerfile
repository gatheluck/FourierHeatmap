FROM nvidia/cuda:11.0-devel-ubuntu20.04 as fhmap_runtime
ARG INSTALL_DIRECTORY=/root/scratch
ARG PYTHON_VERSION=3.9
# restrict log 
ENV PYTHONUNBUFFERED=1

RUN apt update && apt install --no-install-recommends -y \
	git curl ssh openssh-client \
	python${PYTHON_VERSION} python3-pip \
	&& pip3 install poetry

RUN git clone https://github.com/gatheluck/FourierHeatmap.git ${INSTALL_DIRECTORY}/FourierHeatmap
WORKDIR ${INSTALL_DIRECTORY}/FourierHeatmap
RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "fhmap/apps/eval_fhmap.py"]
CMD ["--help"]
