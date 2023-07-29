from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'video_style_transfer': {
		'transforms': transforms_config.StyleTransforms,
		'train_image_root': dataset_paths['image_train'],
		'train_video_root': dataset_paths['video_train'],
		'train_style_root': dataset_paths['style_train'],
		'test_image_root': dataset_paths['image_test'],
		'test_video_root': dataset_paths['video_test'],
		'test_style_root': dataset_paths['style_test'],
	},

}
