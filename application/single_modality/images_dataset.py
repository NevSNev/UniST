from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from utils import data_utils


class ImagesDataset(Dataset):

	def __init__(self, content_root,opts, content_transform):
		self.conent_paths = sorted(data_utils.make_dataset(content_root))
		self.content_transform = content_transform
		self.opts = opts

	def __len__(self):
		return len(self.conent_paths)

	def __getitem__(self, index):
		from_path = self.conent_paths[index]
		from_im = Image.open(from_path).convert('RGB')
		from_im = self.content_transform(from_im)
		return from_im
