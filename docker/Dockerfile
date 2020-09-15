FROM ubuntu:16.04
MAINTAINER <wdg369@gmail.com>

RUN apt-get update
RUN apt-get install -y python3-pip python3-dev
RUN cd /usr/local/bin && \
	ln -s /usr/bin/python3 python && \
	ln -s /usr/bin/pip3 pip && \
	pip3 install --upgrade pip
	
RUN apt-get update && \
    apt-get install -y git

RUN pip3 install -y transformer && \
pip3 install -y sklearn
