#!/bin/bash -xe
# Ensure Python 3 is used
export PYSPARK_PYTHON=/usr/bin/python3
export PYSPARK_DRIVER_PYTHON=/usr/bin/python3

# Install git
sudo yum install git -y


# Non-standard and non-Amazon Machine Image Python modules:
sudo pip install -U \
  boto              \
  tweet-preprocessor

# Install in Python 3
sudo pip-3.4 install -U \
  boto              \
  jupyterlab        \
  findspark         \
  tweet-preprocessor
