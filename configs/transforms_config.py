from abc import abstractmethod
import torchvision.transforms as transforms



class TransformsConfig(object):

	def __init__(self, opts):
		self.opts = opts

	@abstractmethod
	def get_transforms(self):
		pass


class StyleTransforms(TransformsConfig):

	def __init__(self, opts):
		super(StyleTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = transforms.Compose([
			transforms.Resize((512,512)),
			transforms.RandomCrop((256,256)),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		])

		return transforms_dict


