import dna
import hdgim
import torch


# [begin] hyperparameters
HYPERVECTOR_DIMENSION = 1000  # denoted D in paper
DNA_SEQUENCE_LENGTH = 1000
DNA_SUBSEQUENCE_LENGTH = 20  # denoted n in paper
BIT_PRECISION = 2  # denoted B in paper
NOISE_PROBABILTY = 0.1  # denoted p in paper, probability: 0 ~ 1, simulate FeFET noise
# [end] hyperparameters

torch.set_printoptions(sci_mode=False)

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
model.quantize_cdf()
model.noise()

# train
accuracy = model.train(10, 0.1, 0.1, 'cosine', False, True, True)