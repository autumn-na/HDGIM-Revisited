import torch
import dna
from constants import *


class HDGIM:
    def __init__(self):
        self.dna_sequence = None  # 1-dim DNA tensor
        self.dna_chunks = None  # 2-dim DNA tensor
        self.base_hypervectors = None  # dictionary { DNA: tensor }
        self.encoded_hypervector = None  # 2-dim double tensor
        self.quantized_hypervector = None  # 2-dim binary tensor

        self.training_label = None  # 1-dim int tensor, 0: not included | 1: included
        self.training_data = None  # 1-dim DNA tensor

    def create_base_hypervectors(self):
        self.base_hypervectors = {dna.DNA.A: 2 * PI * torch.rand(BASE_DIMENSION) - PI,
                                  dna.DNA.C: 2 * PI * torch.rand(BASE_DIMENSION) - PI,
                                  dna.DNA.G: 2 * PI * torch.rand(BASE_DIMENSION) - PI,
                                  dna.DNA.T: 2 * PI * torch.rand(BASE_DIMENSION) - PI}

    def create_dna_sequence(self):
        self.dna_sequence = dna.create_random_dna_sequence(DNA_SEQUENCE_LENGTH)

    def split_dna_chunks(self):
        self.dna_chunks = dna.get_chunks_dna_sequence(self.dna_sequence, CHUNK_LENGTH)

    def bind(self):
        chunk_hypervectors = None
        for dna_chunk in self.dna_chunks:
            shift_count = 0
            bound_hypervector = None
            chunk_hypervector = None

            for _dna in dna_chunk:
                base_hypervector = torch.unsqueeze(self.base_hypervectors[dna.DNA(_dna.item())], dim=0)

                if bound_hypervector is None:
                    bound_hypervector = base_hypervector
                else:
                    bound_hypervector = torch.concat((bound_hypervector, base_hypervector), dim=0)

            bound_hypervector = torch.roll(bound_hypervector, shifts=shift_count, dims=0)
            shift_count += 1

            if chunk_hypervector is None:
                chunk_hypervector = bound_hypervector
            else:
                chunk_hypervector = torch.mul(chunk_hypervector, bound_hypervector)

            if chunk_hypervectors is None:
                chunk_hypervectors = chunk_hypervector
            else:
                chunk_hypervectors = torch.stack((chunk_hypervectors, chunk_hypervector), dim=0)

        self.encoded_hypervector = torch.sum(chunk_hypervectors, dim=0)

    def quantize(self):
        min_value = torch.min(self.encoded_hypervector)
        max_value = torch.max(self.encoded_hypervector)

        binary_width = (max_value - min_value) / BIT_PRECISION

        self.encoded_hypervector = torch.floor((self.encoded_hypervector + torch.abs(min_value)) / binary_width)

    def create_training_data(self):
        pass
