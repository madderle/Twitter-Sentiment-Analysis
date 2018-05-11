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
  numpy             \
  pandas            \
  tweet-preprocessor

jupyter lab --no-browser --port 7777 --ip='*' --allow-root --NotebookApp.iopub_data_rate_limit=10000000
