# Experiment 2: Distribution of full-precision bound hypervector

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

# draw plot
plt.hist(model.encoded_hypervector.numpy(), bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

plt.title('Distribution of Full-Precision Bound Hypervector')
plt.xlabel('Value')
plt.ylabel('Density')

# Show the plot
plt.show()