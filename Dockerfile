FROM ubuntu:16.04

WORKDIR /working_dir

COPY . /working_dir

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 \
	ca-certificates \
	byobu \
	curl \
	pkg-config \
	python3-dev \
	python3-pip

RUN mkdir -p /working_dir/metadata
RUN mkdir -p /working_dir/images

ENV PYTHONPATH=/working_dir:$PYTHONPATH

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

