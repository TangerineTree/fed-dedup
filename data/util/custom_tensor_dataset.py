from torch.utils.data import Dataset, TensorDataset, ConcatDataset
import numpy as np

#很重要的用来重新组装dataset，在联邦切分数据时，或者划分验证集时要用到
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None, hashes=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors
        self.transform = transform
        self.hashes = hashes

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))

        if self.hashes is not None:
            return x,y,self.hashes[index]
        else:
            return x, y

    def __len__(self):
        return self.tensors[0].size(0)
