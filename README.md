# Estimating Synthetic Data
## Project structure
./  
|- data/		   # data for the project   
|- logs/		   # contains the log file for system logging  
|- notebooks/	   # notebooks for the experiments and implementation  
|- pickles/		   # saved python-objects such as SDG models  
|- src/			   # source code, for help functions and computing PF    
|- tests/		   # test function  
|- Makefile		   # automization of tasks, has some issues in windows, need rework
|- README.md	   # documentation file  
|- config_experiment.yaml  # global settings for the experiment    


## Pre-requesites
- poetry
- python 3.9.13

### Install
Run `poetry install` to dependencies.

### Experiment settings
The file `./config/experiment_settings.yml` contains all the global settings for all runs of the experiment,  
such as the folders for each files, or for choosing how many synthetic datasets to generat, or how large each of   
them should be.