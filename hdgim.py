import torch
import dna
from constants import *


class HDGIM:
    def __init__(self):
        self.dna_sequence = None  # 1-dim DNA tensor
        self.dna_subsequences = None  # 2-dim DNA tensor
        self.base_hypervectors = None  # dictionary { DNA: tensor }
        self.encoded_hypervector = None  # 1-dim double tensor
        self.quantized_hypervector = None  # 1-dim binary tensor

        self.training_label = None  # 1-dim int tensor, 0: not included | 1: included
        self.training_data = None  # 1-dim DNA tensor

    def create_base_hypervectors(self):
        self.base_hypervectors = {dna.DNA.A: 2 * PI * torch.rand(BASE_DIMENSION) - PI,
                                  dna.DNA.C: 2 * PI * torch.rand(BASE_DIMENSION) - PI,
                                  dna.DNA.G: 2 * PI * torch.rand(BASE_DIMENSION) - PI,
                                  dna.DNA.T: 2 * PI * torch.rand(BASE_DIMENSION) - PI}

    def create_dna_sequence(self):
        self.dna_sequence = dna.DNASequence(DNA_SEQUENCE_LENGTH)
        self.dna_sequence.randomize()

    def create_dna_subsequences(self):
        self.dna_subsequences = self.dna_sequence.get_subsequences(CHUNK_LENGTH)

    def bind(self):
        chunk_hypervectors = []
        stacked_chunk_hypervector = torch.empty(len(self.dna_subsequences), BASE_DIMENSION)
    
        for shift_count, dna_subsequence in enumerate(self.dna_subsequences):
            chunk_hypervector = torch.ones(1, BASE_DIMENSION)

            for _dna in dna_subsequence:
                dna_value = dna.DNA(_dna.item())
                base_hypervector = torch.roll(self.base_hypervectors[dna_value], shifts=shift_count, dims=0)
                chunk_hypervector = torch.mul(chunk_hypervector, base_hypervector)

            chunk_hypervectors.append(chunk_hypervector)
  
        stacked_chunk_hypervector = torch.stack(chunk_hypervectors)
        self.encoded_hypervector = torch.squeeze(torch.sum(stacked_chunk_hypervector, dim=0))  # bundling hypervectors

    def quantize_min_max(self):
        min_value = torch.min(self.encoded_hypervector)
        max_value = torch.max(self.encoded_hypervector)

        binary_width = (max_value - min_value) / BIT_PRECISION

        self.quantized_hypervector = torch.floor((self.encoded_hypervector + torch.abs(min_value)) / binary_width)

    def create_training_data(self):
        pass
