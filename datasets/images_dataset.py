from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from utils import data_utils

class ImagesDataset(Dataset):

	def __init__(self, image_root,video_root,content_transform,opts):
		self.image_path = sorted(data_utils.make_dataset(image_root))
		self.video_path = sorted(data_utils.make_dataset(video_root))
		self.content_transform = content_transform
		self.opts = opts
		self.video_root =video_root

	def __len__(self):
		return min(len(self.image_path),len(self.video_path))

	def __getitem__(self, index):
		image = self.image_path[index]
		image = Image.open(image).convert('RGB')
		image = self.content_transform(image)
		video = self.video_path[index]
		video = Image.open(video).convert('RGB')
		video = self.content_transform(video)
		return image,video


