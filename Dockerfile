FROM continuumio/miniconda

MAINTAINER erikreppel@gmail.com

RUN mkdir /work
WORKDIR /work
COPY *.ipynb /work/
COPY env.yml /work/env.yml
COPY experiment.py /work/
COPY utils.py /work/

RUN conda env create -f env.yml

EXPOSE 8888
ENTRYPOINT ['/bin/bash', '-c', 'source activate ml && jupyter notebook --ip=* --no-browser --allow-root']

