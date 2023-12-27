# Experiment 1: Distribution of full-precision base hypervectors

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
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot the histograms in each subplot
axes[0, 0].hist(model.base_hypervectors[dna.DNA.A].numpy(), bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_title('Value of Base Hypervector A')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')

axes[0, 1].hist(model.base_hypervectors[dna.DNA.C].numpy(), bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].set_title('Value of Base Hypervector C')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')

axes[1, 0].hist(model.base_hypervectors[dna.DNA.G].numpy(), bins=30, density=True, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].set_title('Value of Base Hypervector G')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Density')

axes[1, 1].hist(model.base_hypervectors[dna.DNA.T].numpy(), bins=30, density=True, alpha=0.7, color='red', edgecolor='black')
axes[1, 1].set_title('Value of Base Hypervector T')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Density')

# Add a title for the entire figure
plt.suptitle('Distribution of Full-Precision Base Hypervectors')

# Adjust layout to prevent clipping of titles
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()