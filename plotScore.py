import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

from os import listdir
from os.path import isfile, join
import plotly.plotly as py

listFiles = []
scoreNumber = []
y2 = []

#Plot scores for K-means Simple
def kmeanSimple(scores):
	
	for score in scores:
		y2.append(score)

	x2 = [20, 30, 40, 50, 60, 70, 80, 90, 100]
	plt.bar(x2, y2, label='Score', color='blue')
	plt.xlabel('K')
	plt.ylabel('Scores')
	plt.title('K-means simple')
	plt.legend()
	plt.show()

#Plot score for K-means One Hot Encoder
def kmeanOneHotEncoder(scores):

	for score in scores:
		y2.append(score)

	x2 = [20, 30, 40, 50, 60, 70, 80, 90, 100]
	plt.bar(x2, y2, label='Score', color='green')
	plt.xlabel('K')
	plt.ylabel('Scores')
	plt.title('K-means One Hot Encoder')
	plt.legend()
	plt.show()

#Plot scores for K-means One Hot Encoder with normalization
def kmeanOneHotEncoderWithNormalization(scores):

	for score in scores:
		y2.append(score)

	x2 = [20, 30, 40, 50, 60, 70, 80, 90, 100]
	plt.bar(x2, y2, label='Score', color='yellow')
	plt.xlabel('K')
	plt.ylabel('Scores')
	plt.title('K-means One Hot Encoder with normalization')
	plt.legend()
	plt.show()

#Plot score for Bisecting K-means One Hot Encoder with normalization
def bisectingKmeanOneHotEncoderWithNormalization(scores):

	for score in scores:
		y2.append(score)

	x2 = [20, 30, 40, 50, 60, 70, 80, 90, 100]
	plt.bar(x2, y2, label='Score', color='red')
	plt.xlabel('K')
	plt.ylabel('Scores')
	plt.title('Bisecting K-means One Hot Encoder with normalization')
	plt.legend()
	plt.show()

#Plot score Gaussian Mixture One Hot Encoder with normalization
def gaussianMixtureOneHotEncoderWithNormalization(scores):

	for score in scores:
		y2.append(score)

	x2 = [20, 30, 40, 50, 60, 70, 80, 90, 100]
	plt.bar(x2, y2, label='Score', color='orange')
	plt.xlabel('K')
	plt.ylabel('Scores')
	plt.title('Bisecting K-means One Hot Encoder with normalization')
	plt.legend()
	plt.show()

#Read score in file
def readScoreInFile(filename):
	with open(filename) as f:
		lines = f.readlines()
	lines = [line.rstrip('\n') for line in lines]
	#print('lines: '+lines[1])
	lineScore = lines[1].split('=')
	scoreNumber.append(lineScore[1])
	print(filename+' : '+lineScore[1])

#Read files in folder
def readFiles(pathToFolder):

	for f in listdir(pathToFolder):
		if isfile(join(pathToFolder, f)):
			listFiles.append(f)

	for el in listFiles:
		readScoreInFile(pathToFolder+'/'+el)

#Read and plot each technic separately
readFiles('results/kmeans_simple')
kmeanSimple(scoreNumber)

#readFiles('results/kmeans_one_hot_encoder')
#kmeanOneHotEncoder(scoreNumber)

#readFiles('results/kmeans_one_hot_encoder_with_normalization')
#kmeanOneHotEncoderWithNormalization(scoreNumber)

#readFiles('results/bisecting_kmeans_one_hot_encoder_with_normalization')
#bisectingKmeanOneHotEncoderWithNormalization(scoreNumber)

#readFiles('results/gaussian_mixture_one_hot_encoder_with_normalization')
#gaussianMixtureOneHotEncoderWithNormalization(scoreNumber)





		
	










