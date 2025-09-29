from threading import local
import token
import torch
import os
import numpy as np
class FineWebDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path) -> None:
        super().__init__()
        self.files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path)]
        self.file_sizes = []
        self.cumsum_total_sizes = []
        self.cumsum_total = 0
        for f in self.files:
            individual_file = np.load(f, mmap_mode = 'r')
            size = individual_file.shape[0]
            self.file_sizes.append(size)
            self.cumsum_total += size
            self.cumsum_total_sizes.append(self.cumsum_total)



    def __len__(self):
        return self.cumsum_total_sizes[-1]

    def __getitem__(self, idx):
        file_index = np.searchsorted(self.cumsum_total_sizes, idx, side='right')
        local_index = 0
        if file_index == 0:
            local_index = idx
        else:
            local_index = idx - self.cumsum_total_sizes[file_index - 1]

        file = np.load(self.files[file_index], mmap_mode = 'r')
        return torch.from_numpy(file[local_index]).long()    

            

class FineWebDataLoader:
    def __init__(self, folder_path, batch_size, token_length):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        self.batch_size = batch_size
        self.token_length = token_length
        self.current_file_index = 0
        self.current_token_index = 0
        self._load_file()

    def _load_file(self):
        self.tokens = np.load(self.files[self.current_file_index], mmap_mode='r')

    def _load_next_file(self):
        self.current_file_index = (self.current_file_index + 1) % len(self.files)
        self.current_token_index = 0
        self._load_file()

    def _tokens_to_batch(self, tokens):
        x = torch.from_numpy(tokens[:-1]).long().view(self.batch_size, self.token_length)
        y = torch.from_numpy(tokens[1:]).long().view(self.batch_size, self.token_length)
        return x, y

    def next_batch(self):
        tokens_needed = self.batch_size * self.token_length + 1
        remaining = len(self.tokens) - self.current_token_index

        if remaining < tokens_needed:
            # cross-file batch
            batch_tokens = self.tokens[self.current_token_index:]
            self._load_next_file()
            rest_needed = tokens_needed - len(batch_tokens)
            batch_tokens = np.concatenate([batch_tokens, self.tokens[:rest_needed]])
            self.current_token_index = rest_needed
        else:
            batch_tokens = self.tokens[self.current_token_index:self.current_token_index + tokens_needed]
            self.current_token_index += self.batch_size * self.token_length

        return self._tokens_to_batch(batch_tokens)

class FineWebBatchIterator:
    def __init__(self, fw_loader, steps_per_epoch=None):
        self.fw_loader = fw_loader
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        self.step_count = 0
        return self

    def __next__(self):
        if self.steps_per_epoch is not None and self.step_count >= self.steps_per_epoch:
            raise StopIteration
        x, y = self.fw_loader.next_batch()
        self.step_count += 1
        return x, y
