FROM python:3.6.9-stretch

WORKDIR /lfire
ADD . /lfire

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y gfortran graphviz jupyter-notebook

# For some reason glmnet must be installed separately
RUN grep -v 'glmnet' requirements.txt > tmp.txt && mv tmp.txt requirements.txt
RUN pip install --upgrade pip && pip install -e . && pip install glmnet==2.1.1

EXPOSE 8888
