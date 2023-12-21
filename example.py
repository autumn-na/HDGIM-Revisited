import dna
import hdgim
import dna
import torch
from torch.utils.data import Dataset, DataLoader


# [begin] hyperparameters
HYPERVECTOR_DIMENSION = 1000  # denoted D in paper
DNA_SEQUENCE_LENGTH = 100
DNA_SUBSEQUENCE_LENGTH = 10  # denoted n in paper
BIT_PRECISION = 4  # denoted B in paper
NOISE_PROBABILTY = 0.1  # denoted p in paper, probability: 0 ~ 1, simulate FeFET noise
# [end] hyperparameters

torch.set_printoptions(sci_mode=False)

dataset = dna.DNADataset(DNA_SEQUENCE_LENGTH, DNA_SUBSEQUENCE_LENGTH)

model = hdgim.HDGIM(HYPERVECTOR_DIMENSION, DNA_SEQUENCE_LENGTH, DNA_SUBSEQUENCE_LENGTH, BIT_PRECISION, NOISE_PROBABILTY)
model.set_dataset(dataset)
model.create_voltage_matrix()
model.create_base_hypervectors()
model.create_dna_sequence()
model.create_dna_subsequences()
model.bind()
model.quantize_cdf()
model.noise()
model.train(10, 0.1, 0.1)
