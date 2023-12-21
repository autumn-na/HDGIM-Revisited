import dna
import hdgim
import dna
from torch.utils.data import Dataset, DataLoader


# [begin] hyperparameters
HYPERVECTOR_DIMENSION = 1000  # denoted D in paper
DNA_SEQUENCE_LENGTH = 100  # should be divided by DNA_SUBSEQUENCE_LENGTH
DNA_SUBSEQUENCE_LENGTH = 10  # denoted n in paper
BIT_PRECISION = 2  # denoted B in paper
# [end] hyperparameters

dataset = dna.DNADataset(DNA_SEQUENCE_LENGTH, DNA_SUBSEQUENCE_LENGTH)

model = hdgim.HDGIM(HYPERVECTOR_DIMENSION, DNA_SEQUENCE_LENGTH, DNA_SUBSEQUENCE_LENGTH, BIT_PRECISION)
model.set_dataset(dataset)
model.create_voltage_matrix()
model.create_base_hypervectors()
model.create_dna_sequence()
model.create_dna_subsequences()
model.bind()
model.quantize_min_max()
model.noise(0.1)
model.train(10, 0.1, 20)
