import torch
from enum import Enum
from torch.utils.data import Dataset


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
        subsequences = self.dna_sequence.reshape(self.dna_sequence.size(0) // subsequence_length, subsequence_length)
        return subsequences
    
    def get_subsequences_as_dna_sequence(self, subsequence_length):
        result = []
        subsequence_list = list(torch.split(self.dna_sequence, subsequence_length))

        for subsequence in subsequence_list:
            dna_sequence = DNASequence(subsequence_length)
            dna_sequence.set_sequence_by_tensor(subsequence)
            result.append(dna_sequence)
        return result
    
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

        for _ in range(len(self.dna_sequence) // dna_subsequence_length):
            while True:
                false_subsequence = DNASequence(dna_subsequence_length)
                false_subsequence.randomize()

                if not self.dna_sequence.is_contained(false_subsequence):
                    break

            self.false_dna_subsequences.append(false_subsequence)

        self.dna_subsequences = self.true_dna_subsequences + self.false_dna_subsequences

    def __len__(self):
        return len(self.dna_subsequences)

    def __getitem__(self, idx):
        is_contained = self.dna_sequence.is_contained(self.dna_subsequences[idx])
        sample = {'isContained': is_contained, 'subsequence': self.dna_subsequences[idx].get_sequence()}
        return sample
