import torch
import dna
import math
import random
from torch.utils.data import DataLoader

class HDGIM:
    def __init__(self, hypervector_dimension, dna_sequence_length, dna_subsequence_length, bit_precision):
        # [begin] hyperparameters
        self.hypervector_dimension = hypervector_dimension
        self.dna_sequence_length = dna_sequence_length
        self.dna_subsequence_length = dna_subsequence_length
        self.bit_precision = bit_precision
        # [end] hyperparameters

        self.dna_sequence = None  # 1-dim DNA tensor
        self.dna_subsequences = None  # 2-dim DNA tensor
        self.base_hypervectors = None  # dictionary { DNA: tensor }

        self.encoded_hypervector = None  # 1-dim double tensor
        self.encoded_hypervector_library = None  # 2-dim double tensor

        self.quantized_hypervector = None  # 1-dim binary tensor
        self.quantized_hypervector_library = None  # 2-dim binary tensor

        self.noised_quantized_hypervector = None  # 1-dim binary tensor

        self.dna_dataset = None  # DNADataset

    def create_base_hypervectors(self):
        pi = math.pi
        self.base_hypervectors = {dna.DNA.A: 2 * pi * torch.rand(self.hypervector_dimension) - pi,
                                  dna.DNA.C: 2 * pi * torch.rand(self.hypervector_dimension) - pi,
                                  dna.DNA.G: 2 * pi * torch.rand(self.hypervector_dimension) - pi,
                                  dna.DNA.T: 2 * pi * torch.rand(self.hypervector_dimension) - pi}

    def create_dna_sequence(self):
        self.dna_sequence = dna.DNASequence(self.dna_sequence_length)
        self.dna_sequence.randomize()

    def create_dna_subsequences(self):
        self.dna_subsequences = self.dna_sequence.get_subsequences(self.dna_subsequence_length)

    def bind(self):
        chunk_hypervectors = []
    
        for shift_count, dna_subsequence in enumerate(self.dna_subsequences):
            chunk_hypervector = torch.ones(1, self.hypervector_dimension)

            for _dna in dna_subsequence:
                dna_value = dna.DNA(_dna.item())
                base_hypervector = torch.roll(self.base_hypervectors[dna_value], shifts=shift_count, dims=0)
                chunk_hypervector = torch.squeeze(torch.mul(chunk_hypervector, base_hypervector))

            chunk_hypervectors.append(chunk_hypervector)
  
        self.encoded_hypervector_library = torch.stack(chunk_hypervectors)
        self.encoded_hypervector = torch.sum(self.encoded_hypervector_library, dim=0)  # bundling hypervectors

    def bind_dna_sequence(self, dna_sequence):
        chunk_hypervector = torch.ones(1, self.hypervector_dimension)

        for shift_count, _dna in enumerate(dna_sequence):
            dna_value = dna.DNA(_dna.item())
            base_hypervector = torch.roll(self.base_hypervectors[dna_value], shifts=shift_count, dims=0)
            chunk_hypervector = torch.squeeze(torch.mul(chunk_hypervector, base_hypervector))
  
        return chunk_hypervector

    def quantize_min_max(self):
        min_value = torch.min(self.encoded_hypervector)
        max_value = torch.max(self.encoded_hypervector)

        binary_width = (max_value - min_value) / (self.bit_precision + 1)

        self.quantized_hypervector_library = torch.floor((self.encoded_hypervector_library + torch.abs(min_value)) / binary_width)
        self.quantized_hypervector = torch.floor((self.encoded_hypervector + torch.abs(min_value)) / binary_width)

    def quantize_dna_sequence_min_max(self, dna_sequence):
        min_value = torch.min(self.encoded_hypervector)
        max_value = torch.max(self.encoded_hypervector)

        binary_width = (max_value - min_value) / (self.bit_precision + 1)

        return torch.floor((dna_sequence + torch.abs(min_value)) / binary_width)

    def noise(self, probability):
        self.noised_quantized_hypervector = self.quantized_hypervector

        for i, value in enumerate(self.quantized_hypervector):
            is_change = (probability >= random.random())
            if not is_change:
                continue
            
            left_or_right = random.randint(0, 1)
            change_value = -1 if left_or_right == 0 else 1
            noised_value = max(0, min(value + change_value, pow(2, self.bit_precision) - 1))
            self.noised_quantized_hypervector[i] = noised_value

    def set_dataset(self, dna_dataset):
        self.dna_dataset = dna_dataset

    def get_similarity(self, hypervector1, hypervector2):
        difference_hypervector = torch.abs(hypervector1 - hypervector2)
        similarity = 0

        for value in difference_hypervector:
            similarity += torch.sum(self.quantized_hypervector_library[:, int(value.item())], dim=0).item()

        return similarity
    
    def train(self, epoch, learning_rate, threshold):
        data_loader = DataLoader(self.dna_dataset, batch_size=1, shuffle=True)

        for _epoch in range(epoch):
            success_cnt = 0
            for cnt, data in enumerate(data_loader):
                query = torch.squeeze(data['subsequence'])
                encoded_query = self.bind_dna_sequence(query)
                quantized_query = self.quantize_dna_sequence_min_max(encoded_query)

                similarity = self.get_similarity(self.quantized_hypervector, quantized_query)
                divided_similarity = similarity / self.hypervector_dimension

                if (divided_similarity >= threshold) and (data['isContained'].item() == False):
                    self.encoded_hypervector -= learning_rate * encoded_query
                elif (divided_similarity < threshold) and (data['isContained'].item() == True):
                    self.encoded_hypervector += learning_rate * encoded_query
                    success_cnt += 1
                cnt += 1

            print("Epoch {}: Accuracy {}%".format(_epoch, (success_cnt / cnt) * 100))
                


                