FROM continuumio/anaconda3

#Run an update
RUN apt-get update

#Create a user ds
RUN adduser -D -S -h /home/ds -s /bin/ash ds

#install neccessary packages
RUN pip install --upgrade pip
RUN pip install jupyterlab boto3 gensim tweet-preprocessor redis
RUN conda update conda
RUN conda install nltk
RUN conda install spacy -y
RUN conda install -c conda-forge xgboost
RUN python -m spacy download 'en'
RUN python -m nltk.downloader -d /usr/local/share/nltk_data all


#Set Variables
ENV HOME=/home/ds
ENV SHELL=/bin/bash
ENV USER=ds
VOLUME /home/ds
WORKDIR /home/ds

#Add script
ADD Analysis.py /home/ds
