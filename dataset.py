import numpy as np
from glob import glob
from pathlib import Path
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset

class CustomParallelDataset(Dataset):
    def __init__(self, root_dict, loader_dict, extensions_dict, transform=None, cache=False, trim=None) -> None:
        super().__init__()
        self.root_dict = root_dict
        self.loader_dict = loader_dict
        self.transform = transform
        self.cache = cache
        self.trim = 2147483647 if trim is None else trim

        assert root_dict.keys() == self.loader_dict.keys() == extensions_dict.keys()
        self.ks = root_dict.keys()

        self.path_dict = defaultdict(list)
        for k in self.ks:
            for root in root_dict[k]:
                for extension in extensions_dict[k]:
                    self.path_dict[k] += glob(str(Path(root) / ('**/*.' + extension)))
            self.path_dict[k] = sorted(self.path_dict[k])[:self.trim]
        assert len(set([len(self.path_dict[k]) for k in self.path_dict])) <= 1
        self.N = list(set([len(self.path_dict[k]) for k in self.path_dict]))[0]

        # if self.cache:
        #     bank_dict = {}
        #     for (k, paths), (_, load_fn) in zip(self.path_dict.items(), self.loader_dict.items()):
        #         for idx, path in enumerate(paths):
        #             if k in bank_dict:
        #                 bank_dict[k][idx] = load_fn(path)
        #             else:
        #                 arr = load_fn(path)
        #                 import pdb; pdb.set_trace()
        #                 bank_dict[k] = np.zeros(self.N, *(arr.shape))
        #     self.bank_dict = bank_dict

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        # if self.cache:
        #     out = {k: self.bank_dict[k][index] for k in self.ks}
        # else:
        #     out = {k: load_fn(paths[index]) for (k, paths), (_, load_fn)
        #             in zip(self.path_dict.items(), self.loader_dict.items())}

        out = {k: load_fn(paths[index]) for (k, paths), (_, load_fn)
                    in zip(self.path_dict.items(), self.loader_dict.items())}

        if self.transform:
            out = {k: self.transform(out[k]) for k in out}

        return out