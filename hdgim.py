import torch
import dna
import math
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
import scipy.stats as stats
import numpy as np

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
        sorted_tensor, indices = torch.sort(self.encoded_hypervector)
        np_sorted_tensor = sorted_tensor.numpy()
        np_normalized_sorted_tensor = (np_sorted_tensor - np_sorted_tensor.mean()) / np_sorted_tensor.std()
        binary_width = 1.0 / 2**self.bit_precision

        rv = stats.norm(0, 1) # assume standard normal distribution
        cdf = rv.cdf(np_normalized_sorted_tensor)
        quantized_cdf = np.floor(cdf / binary_width)
        self.quantized_hypervector = torch.from_numpy(quantized_cdf)[torch.argsort(indices)]

    def quantize_dna_sequence_min_max(self, dna_sequence):
        min_value = torch.min(dna_sequence)
        max_value = torch.max(dna_sequence)

        binary_width = (max_value - min_value) / (self.bit_precision + 1)

        return torch.floor((dna_sequence + torch.abs(min_value)) / binary_width)
    
    def quantize_dna_sequence_cdf(self, dna_sequence):
        sorted_tensor, indices = torch.sort(dna_sequence)
        np_sorted_tensor = sorted_tensor.numpy()
        np_normalized_sorted_tensor = (np_sorted_tensor - np_sorted_tensor.mean()) / np_sorted_tensor.std()
        binary_width = 1.0 / 2**self.bit_precision

        rv = stats.norm(0, 1) # assume standard normal distribution
        cdf = rv.cdf(np_normalized_sorted_tensor)
        quantized_cdf = np.floor(cdf / binary_width)

        return torch.from_numpy(quantized_cdf)[torch.argsort(indices)]

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

    def get_similarity_by_voltage_matrix(self, hypervector1, hypervector2):  # 0 ~ 1000
        distance = 0

        for i in range(self.hypervector_dimension):
           voltage = self.voltage_matrix[int(hypervector1[i].item())][int(hypervector2[i].item())]
           distance += voltage

        return -distance
    
    def get_similarity_by_hamming_distance(self, hypervector1, hypervector2):
        # Calculate Hamming distance element-wise
        # distance = 0

        # for i in range(self.hypervector_dimension):
        #     item1 = int(hypervector1[i].item())
        #     item2 = int(hypervector2[i].item())

        #     distance += bin(item1 ^ item2).count('1')
            
        # return -distance

        return -torch.sum(torch.abs(hypervector1 - hypervector2))
    
    def get_similarity_by_euclidean_distance(self, hypervector1, hypervector2):  # 0 ~ 1
        return -torch.dist(hypervector1, hypervector2, 2)
    
    def get_similarity_by_cosine_similarity(self, hypervector1, hypervector2):  # -1 ~ 1
        return F.cosine_similarity(hypervector1, hypervector2, dim=0)
    
    def train(self, epoch, learning_rate, threshold, f='voltage', full_precision=False, return_data=False, print_info=False):
        train_dataset = self.dna_dataset
        train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        accuracies = []
        true_similarities = []
        false_similarities = []

        if print_info:
            print("Train size: {}".format(len(train_dataset)))

        similarity_function = None
        if f == 'voltage':
            similarity_function = self.get_similarity_by_voltage_matrix
        elif f == 'hamming':
            similarity_function = self.get_similarity_by_hamming_distance
        elif f == 'euclidean':
            similarity_function = self.get_similarity_by_euclidean_distance
        elif f == 'cosine':
            similarity_function = self.get_similarity_by_cosine_similarity

        # train
        for _epoch in range(epoch):
            success_cnt = 0
            true_negative_cnt = 0
            true_positive_cnt = 0   
            false_negative_cnt = 0  
            false_positive_cnt = 0 

            true_similarities.append([])
            false_similarities.append([])

            for i, data in enumerate(train_data_loader):
                query = torch.squeeze(data['subsequence'])
                encoded_query = self.bind_dna_sequence(query)
                quantized_query = self.quantize_dna_sequence_cdf(encoded_query)

                similarity = 0
                divided_similarity = 0

                if full_precision:
                    similarity = similarity_function(self.encoded_hypervector, encoded_query)
                    divided_similarity = similarity
                else:
                    similarity = similarity_function(self.noised_quantized_hypervector, quantized_query)
                    divided_similarity = similarity / self.hypervector_dimension

                is_contained = data['isContained'].item()

                if (divided_similarity < threshold) and (is_contained is False):  # true negative
                    true_negative_cnt += 1
                    success_cnt += 1
                elif (divided_similarity >= threshold) and (is_contained is True):  # true positive
                    true_positive_cnt += 1
                    success_cnt += 1
                elif (divided_similarity >= threshold) and (is_contained is False):  # false negative
                    self.encoded_hypervector -= learning_rate * encoded_query
                    self.quantize_cdf()
                    self.noise()
                    false_negative_cnt += 1
                elif (divided_similarity < threshold) and (is_contained is True):  # false positive
                    self.encoded_hypervector += learning_rate * encoded_query
                    self.quantize_cdf()
                    self.noise()
                    false_positive_cnt += 1

                if is_contained is False:
                    false_similarities[_epoch].append(divided_similarity)
                else:
                    true_similarities[_epoch].append(divided_similarity)

            accuracy = round((success_cnt / len(train_data_loader)) * 100, 2)
            accuracies.append(accuracy)

            if print_info:
                print("Epoch {}: Accuracy {}%".format(_epoch, accuracy))
                print("Average true similarity: {}, Average false similarity: {}".format(sum(true_similarities[_epoch]) / len(true_similarities[_epoch]), sum(false_similarities[_epoch]) / len(false_similarities[_epoch])))
                print("True negative: {}, True positive: {}, False negative: {}, False positive: {}".format(true_negative_cnt, true_positive_cnt, false_negative_cnt, false_positive_cnt))

        if return_data:
            return accuracies, true_similarities, false_similarities
