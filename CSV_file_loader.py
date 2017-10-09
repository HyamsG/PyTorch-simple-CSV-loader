import torch
import torch.utils.data as data
import pandas as pd

class CustomDataset(data.Dataset):

    def __init__(self,path, transforms=None):
        self.X, self.Y = self.load_data_to_tensors(path)
        self.transforms = transforms

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.X)

    def data_loader(self, batch_size):
        return data.DataLoader(dataset=self, batch_size = batch_size)

    def load_data_to_tensors(self,path):
        def custom_manioulation_function(row):
            '''this one is implemented for the heart.csv file from stanford-tensorflow-tutorials'''
            if row[4] == 'Present':
                row[4] = 1
            else:
                row[4] = 0
            return

        DataFrame = pd.read_csv(path)
        X, Y = list(), list()
        for index, row in DataFrame.iterrows():
            custom_manioulation_function(row)
            X.append(torch.Tensor(row[:-1]))
            Y.append(int(row[-1]))
        X = torch.stack(X)
        Y = torch.LongTensor(Y) # may change, depends on the model
        return X, Y

def usege_example(path):
    dataset = CustomDataset(path)
    data = dataset.data_loader(4) #batch size of 4
    for item in enumerate(data):
        print(item)
