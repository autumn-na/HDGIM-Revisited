import torch
from enum import Enum
from torch.utils.data import Dataset
from constants import *


class DNA(Enum):
    A = 0
    C = 1
    G = 2
    T = 3


class DNADataset(Dataset):
    def __init__(self, dna_sequence):
        self.dna_sequence = dna_sequence
        self.true_chunks = list(torch.split(self.dna_sequence, CHUNK_LENGTH))
        self.false_chunks = []
        self.chunks = []

        for i in range(len(dna_sequence) // CHUNK_LENGTH):
            while True:
                tmp_false_chunk = create_random_dna_sequence(CHUNK_LENGTH)
                is_contained = is_dna_chunk_contained(self.dna_sequence, tmp_false_chunk)

                if not is_contained:
                    break

            self.false_chunks.append(tmp_false_chunk)

        self.chunks = self.true_chunks + self.false_chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        is_contained = is_dna_chunk_contained(self.dna_sequence, self.chunks[idx])
        sample = {'isContained': is_contained, 'chunk': self.chunks[idx]}
        return sample


def create_random_dna_sequence(length):
    dna_values = list(DNA)
    dna_sequence = torch.randint(len(dna_values), size=(length,))

    return dna_sequence


def get_chunks_dna_sequence(dna_sequence, chunk_size):
    chunks = dna_sequence.reshape(dna_sequence.size(0) // chunk_size, chunk_size)
    return chunks


def is_dna_chunk_contained(dna_sequence, dna_chunk):
    len_dna_sequence = dna_sequence.size(0)
    len_dna_chunk = dna_chunk.size(0)

    for i in range(len_dna_sequence - len_dna_chunk + 1):
        if torch.equal(dna_sequence[i:i + len_dna_chunk], dna_chunk):
            return True

    return False
