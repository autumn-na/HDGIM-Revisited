# Experiment 6: Similarity Distribution of full-precision bound hypervector

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dna
import hdgim
import torch
import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)

# [begin] hyperparameters
HYPERVECTOR_DIMENSION = 1000  # denoted D in paper
DNA_SEQUENCE_LENGTH = 1000
DNA_SUBSEQUENCE_LENGTH = 20  # denoted n in paper
BIT_PRECISION = 2  # denoted B in paper
NOISE_PROBABILTY = 0.1  # denoted p in paper, probability: 0 ~ 1, simulate FeFET noise
EPOCHES = 10
FUNCTION = 'cosine'
LEARNING_RATE = 10000
THRESHOLD = -0.07
# [end] hyperparameters

# create dataset
dataset = dna.DNADataset(DNA_SEQUENCE_LENGTH, DNA_SUBSEQUENCE_LENGTH)

# create model
model = hdgim.HDGIM(HYPERVECTOR_DIMENSION, DNA_SEQUENCE_LENGTH, DNA_SUBSEQUENCE_LENGTH, BIT_PRECISION, NOISE_PROBABILTY)
model.set_dataset(dataset)
model.create_voltage_matrix()
model.create_base_hypervectors()
model.create_dna_sequence()
model.create_dna_subsequences()
model.bind()
model.quantize_min_max()
model.noise()

# train
accuracies, true_similarities, false_similarities = model.train(EPOCHES, LEARNING_RATE, THRESHOLD, FUNCTION, True, True, True)

# draw histogram which shows true similarity distribution and false similarity distribution
plt.hist(true_similarities[EPOCHES-1], bins=30, density=True, alpha=0.5, color='blue', edgecolor='black', label='true')
plt.hist(false_similarities[EPOCHES-1], bins=30, density=True, alpha=0.5, color='red', edgecolor='black', label='false')

# add explanation of color
plt.legend(prop={'size': 10})

plt.title('Similarity Distribution Graph')
plt.xlabel('Similarity')
plt.ylabel('Density')
plt.show()
