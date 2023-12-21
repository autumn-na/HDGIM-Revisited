import torch
import dna
import math
import random
from torch.utils.data import DataLoader, random_split

class HDGIM:
    def __init__(self, hypervector_dimension, dna_sequence_length, dna_subsequence_length, bit_precision, noise_probability):
        # [begin] hyperparameters
        self.hypervector_dimension = hypervector_dimension
        self.dna_sequence_length = dna_sequence_length
        self.dna_subsequence_length = dna_subsequence_length
        self.bit_precision = bit_precision
        self.noise_probability = noise_probability
        # [end] hyperparameters

        self.max_value = pow(2, self.bit_precision) - 1  # max value of quantized hypervector
        self.voltage_matrix = None  # (max_value + 1)-dim int tensor, denoted M^c in paper

        self.dna_sequence = None  # 1-dim DNA tensor
        self.dna_subsequences = None  # 2-dim DNA tensor
        self.base_hypervectors = None  # dictionary { DNA: tensor }

        self.encoded_hypervector = None  # 1-dim double tensor
        self.encoded_hypervector_library = None  # 2-dim double tensor

        self.quantized_hypervector = None  # 1-dim binary tensor
        self.quantized_hypervector_library = None  # 2-dim binary tensor

        self.noised_quantized_hypervector = None  # 1-dim binary tensor

        self.dna_dataset = None  # DNADataset

    def create_voltage_matrix(self):
        self.voltage_matrix = torch.ones(self.max_value + 1, self.max_value + 1)
        self.voltage_matrix.fill_diagonal_(0)

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

    def quantize_cdf(self):
        cdf_values = 0.5 * (1 + torch.erf(self.encoded_hypervector / math.sqrt(2)))
        quantization_values = torch.linspace(0, 1, self.bit_precision + 1)
        self.quantized_hypervector = torch.searchsorted(quantization_values, cdf_values)

    def quantize_dna_sequence_min_max(self, dna_sequence):
        min_value = torch.min(self.encoded_hypervector)
        max_value = torch.max(self.encoded_hypervector)

        binary_width = (max_value - min_value) / (self.bit_precision + 1)

        return torch.floor((dna_sequence + torch.abs(min_value)) / binary_width)

    # Assume that left probability is same as right probability
    def noise(self):
        self.noised_quantized_hypervector = self.quantized_hypervector

        for i, value in enumerate(self.quantized_hypervector):
            is_change = (self.noise_probability > random.random())
            if not is_change:
                continue
          
            left_or_right = 0  # 0: left, 1: right
            value_int = value.item()

            if value_int == 0:
                left_or_right = 1
            elif value_int == self.max_value:
                left_or_right = 0
            else:
                left_or_right = random.randint(0, 1)

            change_value = -1 if left_or_right == 0 else 1
            noised_value = value_int + change_value
            self.noised_quantized_hypervector[i] = noised_value
    
    def noise_dna_sequence(self, dna_sequence):
        for i, value in enumerate(self.dna_sequence.get_sequence()):
            is_change = (self.noise_probability > random.random())
            if not is_change:
                continue
          
            left_or_right = 0  # 0: left, 1: right
            value_int = value.item()

            if value_int == 0:
                left_or_right = 1
            elif value_int == self.max_value:
                left_or_right = 0
            else:
                left_or_right = random.randint(0, 1)

            change_value = -1 if left_or_right == 0 else 1
            noised_value = value_int + change_value
            dna_sequence[i] = noised_value
        
        return dna_sequence

    def set_dataset(self, dna_dataset):
        self.dna_dataset = dna_dataset

    def get_similarity(self, hypervector1, hypervector2):  # 0 ~ 1000
        similarity = 0

        for i in range(self.hypervector_dimension):
           voltage = self.voltage_matrix[int(hypervector1[i].item())][int(hypervector2[i].item())]
           similarity += voltage

        return similarity
    
    def train(self, epoch, learning_rate, threshold):
        train_size = int(0.8 * len(self.dna_dataset))
        test_size = len(self.dna_dataset) - train_size

        train_dataset, test_dataset = random_split(self.dna_dataset, [train_size, test_size])
        print("Train size: {}, Test size: {}".format(train_size, test_size))

        train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        # train
        for _epoch in range(epoch):
            for data in train_data_loader:
                query = torch.squeeze(data['subsequence'])
                encoded_query = self.bind_dna_sequence(query)
                quantized_query = self.quantize_dna_sequence_min_max(encoded_query)
                noised_quantized_query = self.noise_dna_sequence(quantized_query)

                similarity = self.get_similarity(self.noised_quantized_hypervector, noised_quantized_query)
                divided_similarity = similarity / self.hypervector_dimension  # 0 ~ 1

                if (divided_similarity >= threshold) and (data['isContained'].item() == False):
                    self.encoded_hypervector -= learning_rate * encoded_query
                elif (divided_similarity < threshold) and (data['isContained'].item() == True):
                    self.encoded_hypervector += learning_rate * encoded_query

            # test
            success_cnt = 0

            # test values
            true_negative_cnt = 0
            true_positive_cnt = 0
            false_negative_cnt = 0
            false_positive_cnt = 0
            
            for data in test_data_loader:
                query = torch.squeeze(data['subsequence'])
                encoded_query = self.bind_dna_sequence(query)
                quantized_query = self.quantize_dna_sequence_min_max(encoded_query)
                noised_quantized_query = self.noise_dna_sequence(quantized_query)

                similarity = self.get_similarity(self.noised_quantized_hypervector, noised_quantized_query)
                divided_similarity = similarity / self.hypervector_dimension

                if (divided_similarity >= threshold) and (data['isContained'].item() == False):  # true negative
                    true_negative_cnt += 1
                    success_cnt += 1
                elif (divided_similarity < threshold) and (data['isContained'].item() == True):  # true positive
                    true_positive_cnt += 1
                    success_cnt += 1
                elif (divided_similarity < threshold) and (data['isContained'].item() == False):  # false negative
                    false_negative_cnt += 1
                elif (divided_similarity >= threshold) and (data['isContained'].item() == True):  # false positive
                    false_positive_cnt += 1

            print("Epoch {}: Accuracy {}%".format(_epoch, round((success_cnt / len(test_data_loader)) * 100, 2)))
            print("True negative: {}, True positive: {}, False negative: {}, False positive: {}".format(true_negative_cnt, true_positive_cnt, false_negative_cnt, false_positive_cnt))
                