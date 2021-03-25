import pandas as pd
import os
import os.path
import torch
from torch.utils.data import Dataset


class AUDataset(Dataset):
    """
    Klasse zum Laden der CSV-Dateien
    Wird für den Expression-Generator benötigt und ist von Pytorch vorgesehen
    """
    def define_longest_sequence(self):
        self.max_sequence_length = 0
        for one_csv in self.csv_names_list:
            df = pd.read_csv(self.csv_directory + "/" + one_csv)
            csv_length = len(df)

            if self.max_sequence_length < csv_length:
                self.max_sequence_length = csv_length
            else:
                self.max_sequence_length = self.max_sequence_length

        return self.max_sequence_length,


    def __init__(self, csv_directory, transform=None):
        """
        :param csv_directory (string): Path to csv-file directory
        :param transform: transforms applied on sample
        """
        self.csv_directory = csv_directory
        self.csv_names_list = os.listdir(self.csv_directory)
        self.transform = transform
        self.all_data_tensor_list = []
        # clean up the list (only use "filled"-files
        for csv in self.csv_names_list:
            if not "_" in csv:
                self.csv_names_list.remove(csv)

        # number of sequences is amount of correct csv_files
        self.number_of_sequences = len(self.csv_names_list)
        self.max_sequence_length = self.define_longest_sequence()


    def __getitem__(self, idx):
        current_sequence = self.csv_names_list[idx]
        df = pd.read_csv(self.csv_directory + "/" + current_sequence)
        csv_data = df.iloc[0:len(df), 1:].values
        sequence_tensor = torch.tensor(csv_data, dtype=torch.float32)

        # print("in get item:", sequence_tensor.type())

        # evtl quatsch
        if self.transform is not None:
            sequence_tensor = self.transform(sequence_tensor)

        # no neg values
        sequence_tensor = torch.clamp(sequence_tensor, min=0.00, max=1.25)
        # round to 4 decimals!
        n_digits = 4
        sequence_tensor = (sequence_tensor * 10**n_digits).round() / (10**n_digits)

        self.all_data_tensor_list.append(sequence_tensor)
        return sequence_tensor, str(current_sequence)

    def __len__(self):
        return self.number_of_sequences


class PadSequencer:
    def __call__(self, data):
        sorted_data = sorted(data, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_data]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.LongTensor([len(x) for x in sequences])
        names = [x[1] for x in sorted_data]
        return sequences_padded, lengths, names


