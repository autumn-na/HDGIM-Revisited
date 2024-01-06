import torch
from enum import Enum
from torch.utils.data import Dataset
import random


class DNA(Enum):
    A = 0
    C = 1
    G = 2
    T = 3

class DNASequence:
    def __init__(self, length):
        self.length = length
        self.dna_sequence = None  # Tensor

    def __len__(self):
        return self.length

    def randomize(self):
        dna_values = list(DNA)
        self.dna_sequence = torch.randint(len(dna_values), size=(self.length,))

    def get_sequence(self):
        return self.dna_sequence

    def get_subsequences(self, subsequence_length):
        subsequences = []

        for i in range(self.length - subsequence_length + 1):
            subsequences.append(self.dna_sequence[i:i + subsequence_length])

        return torch.stack(subsequences)
    
    def get_subsequences_as_dna_sequence(self, subsequence_length):
        subsequences = []

        for i in range(self.length - subsequence_length + 1):
            dna_sequence = DNASequence(subsequence_length)
            dna_sequence.set_sequence_by_tensor(self.dna_sequence[i:i + subsequence_length])
            subsequences.append(dna_sequence)
        
        return subsequences
    
    def is_contained(self, dna_sequence):
        length = len(dna_sequence)

        for i in range(self.length - length + 1):
            if torch.equal(self.dna_sequence[i:i + length], dna_sequence.get_sequence()):
                return True

        return False
    
    def set_sequence_by_tensor(self, tensor):
        self.dna_sequence = tensor

class DNADataset(Dataset):
    def __init__(self, dna_sequence_length, dna_subsequence_length):
        self.dna_sequence = DNASequence(dna_sequence_length)
        self.dna_sequence.randomize()

        self.true_dna_subsequences = self.dna_sequence.get_subsequences_as_dna_sequence(dna_subsequence_length)
        self.false_dna_subsequences = []
        self.dna_subsequences = []

        for _ in range(len(self.true_dna_subsequences)):
            while True:
                false_subsequence = DNASequence(dna_subsequence_length)
                false_subsequence.randomize()

                if not self.dna_sequence.is_contained(false_subsequence):
                    break

            self.false_dna_subsequences.append(false_subsequence)

        self.dna_subsequences = self.true_dna_subsequences + self.false_dna_subsequences
        random.shuffle(self.dna_subsequences)

    def __len__(self):
        return len(self.dna_subsequences)

    def __getitem__(self, idx):
        is_contained = self.dna_sequence.is_contained(self.dna_subsequences[idx])       
        sample = {'isContained': is_contained, 'subsequence': torch.squeeze(self.dna_subsequences[idx].get_sequence())}
        return sample
