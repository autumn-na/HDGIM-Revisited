import dna
import hdgim
import dna
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


# [begin] hyperparameters
HYPERVECTOR_DIMENSION = 1000  # denoted D in paper
DNA_SEQUENCE_LENGTH = 10
DNA_SUBSEQUENCE_LENGTH = 2  # denoted n in paper
BIT_PRECISION = 2  # denoted B in paper
NOISE_PROBABILTY = 0.1  # denoted p in paper, probability: 0 ~ 1, simulate FeFET noise
# [end] hyperparameters

torch.set_printoptions(sci_mode=False)

# experiment settings
dimension_of_hypervectors = [i * 500 for i in range(1, 21)]
probability_of_error = [i / 10 for i in range(11)]
accuracy_matrix = np.zeros((11, 20))

for i, dimension in enumerate(dimension_of_hypervectors):
    for j, probability in enumerate(probability_of_error):
        # create dataset
        dataset = dna.DNADataset(DNA_SEQUENCE_LENGTH, DNA_SUBSEQUENCE_LENGTH)

        # create model
        model = hdgim.HDGIM(dimension, DNA_SEQUENCE_LENGTH, DNA_SUBSEQUENCE_LENGTH, BIT_PRECISION, probability)
        model.set_dataset(dataset)
        model.create_voltage_matrix()
        model.create_base_hypervectors()
        model.create_dna_sequence()
        model.create_dna_subsequences()
        model.bind()
        model.quantize_cdf()
        model.noise()

        # train
        accuracy = model.train(1, 0.1, 0.1, True, False)

        accuracy_matrix[j][i] = accuracy
        # print('dimension: {}, probability: {}, accuracy: {}'.format(i, j, accuracy))

# create discrete colormap which is #FF0000 to #FFFFFF, change green and blue simultaneously
cmap = colors.ListedColormap(['#FF0000', '#FF1A1A', '#FF3333', '#FF4D4D', '#FF6666', '#FF8080', '#FF9999', '#FFB3B3', '#FFCCCC', '#FFE6E6', '#FFFFFF'])
bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
ax.imshow(accuracy_matrix, cmap=cmap, norm=norm)

# create a list of string, which is the labels of the grid, 500 to 10000, step 500, but label is empty if not divided by 2000
labels = []
for i in range(1, 21):
    if i % 4 == 0:
        labels.append(str(i * 500))
    else:
        labels.append('')

# create a list of string, which is the labels of the grid, 0% to 100%, step 10%, but label is empty if not divided by 20%
labels2 = []
for i in range(0, 11):
    if i % 2 == 0:
        labels2.append(str(i * 10) + '%')
    else:
        labels2.append('')

# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
plt.xticks(np.arange(0.5, accuracy_matrix.shape[1], 1), labels) 
plt.yticks(np.arange(0.5, accuracy_matrix.shape[0], 1), labels2)

plt.title('Accuracy Matrix, 2-bit Precision, 10% Noise')
plt.xlabel('Dimension of Hypervectors')
plt.ylabel('Probability of Error')

plt.show()