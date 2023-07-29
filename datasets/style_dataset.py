from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class StyleDataset(Dataset):

    def __init__(self,style_root,opts,style_transform):
        self.style_paths = sorted(data_utils.make_dataset(style_root))
        self.style_transform = style_transform
        self.opts = opts

    def __len__(self):
        return len(self.style_paths)

    def __getitem__(self, index):
        to_path = self.style_paths[index]
        to_im = Image.open(to_path).convert('RGB')
        to_im = self.style_transform(to_im)
        return to_im
