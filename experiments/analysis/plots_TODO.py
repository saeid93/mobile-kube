# loads stuff in the experiments folder to generate
# the final output
import os
import sys
import pandas as pd
import json
import pickle

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    EXPERIMENTS_PATH,
    PLOTS_PATH
)

experiment_id = 1
this_experiment_folder = os.path.join(
	EXPERIMENTS_PATH, str(
		experiment_id))
# save the necesarry information
with open(
os.path.join(
	this_experiment_folder, 'info.json')) as in_file:
	info = json.loads(in_file.read())
with open(
os.path.join(
	this_experiment_folder, 'episodes.pickle'), 'rb') as in_pickle:
	episodes = pickle.load(in_pickle)

# TODO write the rest here