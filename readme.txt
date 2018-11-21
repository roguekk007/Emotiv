This is the work folder containing source code for paper:
	"Utilizing deep learning models for classification of motor-imagery EEG signals"
Written by XingJian Lyu, mentored by Meng Tang

The project is written in Visual Studio Code, python version 3.6, and pytorch version 0.4

"data" folder contains original data (emotiv.mat) and source code for loading and sampling data
	The revised sliding window temporal data sampling method is contained in data.py

"evaluation" contains evaluation.py, and describes utilities for evaluating models on torch DataLoader

"models" contains the four base deep learning models mentioned in the paper.
	All models contain layer_output function which extracts features from the given layer.
	For models with parallel architecture the layer_output function was designed to return feature
	from certain layers irrespective of input

"saved features" contains cached features generated for analyzing improved performance of machine
learning algorithms.

"Emotiv data analysis.ipynb" is a jupyter notebook which contains the figures mentioned in data analysis

"feature_improvement.py" is the script which analyzes various machine learning algorithms on features

"training.py" trains and saves the four base deep learning models

"visualize_performance.py" generates figures visualizing the deep learning models' performance 

The folder also contains saved models which were analyzed in the paper, a suffix is added after their 
file names
Options are provided in feature_improvement.py and visualize_performance.py to generate the exact 
same graphs in the paper 

The hybrid model fits precariously into Nvidia Tesla K80. 

To replicate results:
	Set mode in training.py into 'replicate' and mode in visualize_performance.py into 'new_saved_models'

*Random seeds are set for most procedures, but the author cannot guarantee complete replication

The author promises that codes are the results of the author's own work and effort guided by mentor

