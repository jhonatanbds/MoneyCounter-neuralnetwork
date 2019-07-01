FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python 
RUN apt-get -y install libsm6 libxext6 libxrender-dev
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
CMD ["python3", "run_keras_server.py"]