#!/bin/bash

# Store the name of a conda virtual environment
echo "Input the name of a conda virtual environment for test codes: "
read conda_env_name

echo "The name of rhe conda virtual environment: $conda_env_name"

#conda update -n base conda -y
#conda update --all -y
#python -m pip install --upgrade pip 

#conda create --name $conda_env_name --file requirements_test.txt -c conda-forge -c pytorch -c nvidia -c jmcmurray -y

conda deactivate

source ~/anaconda3/etc/profile.d/conda.sh

echo "ph01"
conda activate
conda activate $conda_env_name

echo "ph02"
#pip3 install wget