from argparse import ArgumentParser

class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', default='', type=str, help='experiment_path')
		self.parser.add_argument('--dataset_type', default='video_style_transfer', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--batch_size', default = 3, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=3, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=4, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.00005, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')

		self.parser.add_argument('--id_1_lambda', default=0.1, type=float, help='ID_1 loss multiplier factor')
		self.parser.add_argument('--id_2_lambda', default=0.5, type=float, help='ID_2 loss multiplier factor')
		self.parser.add_argument('--style_lambda', default=1.5, type=float, help='style loss multiplier factor')
		self.parser.add_argument('--content_lambda', default=0.1, type=float, help='content loss multiplier factor')
		self.parser.add_argument('--temporary_lambda', default=90, type=float, help='cos loss multiplier factor')

		self.parser.add_argument('--video_path_train', default='', type=str, help='Path to directory of videos training set')
		self.parser.add_argument('--image_path_train', default='', type=str, help='Path to directory of images training set')
		self.parser.add_argument('--style_path_train', default='', type=str, help='Path to directory of style training set')
		self.parser.add_argument('--video_path_test', default='', type=str, help='Path to directory of videos test set')
		self.parser.add_argument('--image_path_test', default='', type=str, help='Path to directory of images test set')
		self.parser.add_argument('--style_path_test', default='', type=str, help='Path to directory of style test set')

		self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=10000, type=int, help='Model checkpoint interval')

	def parse(self):
		opts = self.parser.parse_args()
		return opts