# ===========================================
# ||                                       ||
# ||       Section 1: Importing modules    ||
# ||                                       ||
# ===========================================


import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from torch.utils.data import Dataset


# ===========================================
# ||                                       ||
# ||       Section 1: DataModule           ||
# ||                                       ||
# ===========================================

class TensorizedDigits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode = "train", transforms = None, tensorized = True):
        digits = load_digits()
        if mode == "train":
            self.data = digits.data[:1000].astype(np.float32)
            self.targets = digits.target[:1000]
        elif mode == "val":
            self.data = digits.data[1000:1350].astype(np.float32)
            self.targets = digits.target[1000:1350]
        else:
            self.data = digits.data[1350:].astype(np.float32)
            self.targets = digits.target[1350:]

        self.transforms = transforms

        if tensorized:
          self.transforms = TensorizedDigits.tensorization_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_x = self.data[idx]
        sample_y = self.targets[idx]
        
        if True:
          sample_x, sample_y = self.transforms(sample_x, sample_y)
        

        return (sample_x, sample_y)

    @staticmethod
    def tensorization_transform(x, y):
        
        # reshape to get a valid input for a CNN
        sample_x = x.reshape(1, 8, 8)
        sample_y = y

        # transform it to torch tensor to move them to cuda
        if torch.cuda.is_available():

          sample_x = torch.from_numpy(sample_x).to("cuda")
          sample_y = np.array(y)
          sample_y = torch.from_numpy(sample_y).to("cuda")


        return sample_x, sample_y
    
    def visualize_datapoint(self, idx):

      x,y = self.__getitem__( idx)
      plt.imshow(x[0].cpu(), cmap="gray")
      plt.axis("off")
      plt.show()
