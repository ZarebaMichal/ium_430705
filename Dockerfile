FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y figlet

WORKDIR /app

COPY ./requirements.txt ./
COPY ./script2.py ./

RUN pip3 install -r ./requirements.txt
CMD python3 script2.py
CMD python3 lab05.py
CMD python3 lab06_training.py