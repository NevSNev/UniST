from argparse import ArgumentParser

class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir',  default='',type=str, help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default='.pt', type=str, help='Path to UniST model checkpoint')
		self.parser.add_argument('--video_path', type=str, default='', help='Path to directory of content videos')
		self.parser.add_argument('--image_path', type=str, default='', help='Path to directory of content images')
		self.parser.add_argument('--style_image_path', type=str, default='', help='Path to directory of style images to image style transfer')
		self.parser.add_argument('--style_video_path', type=str, default='', help='Path to directory of style images to video style transfer')
		self.parser.add_argument('--content_size', type=int, default=256, help='The size of the images')
		self.parser.add_argument('--style_size', type=int, default=256, help='The size of the images')
		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=4, type=int, help='Number of test/inference dataloader workers')

	def parse(self):
		opts = self.parser.parse_args()
		return opts

	