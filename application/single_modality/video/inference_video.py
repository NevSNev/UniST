import torchvision.transforms as transforms
import os
import numpy as np
import torch
import sys
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
sys.path.append(".")
sys.path.append("..")
from torch.utils.data import DataLoader
from argparse import Namespace
from tqdm import tqdm
from utils.common import tensor2im
from options.test_options import TestOptions
from models.model import video_Style_transfer
from datasets.images_dataset import ImagesDataset

def run():
    test_opts = TestOptions().parse()
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    os.makedirs(out_path_results, exist_ok=True)
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    net = video_Style_transfer(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    transforms_content = transforms.Compose([
        transforms.Resize((opts.content_size, opts.content_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    content_dataset = ImagesDataset(content_root=opts.content_path,
                                    content_transform=transforms_content,
                                    opts=opts)

    content_dataloader = DataLoader(content_dataset,
                                    batch_size=opts.test_batch_size,
                                    shuffle=False,
                                    num_workers=int(opts.test_workers),
                                    drop_last=True)


    print("Number of training samples: {}".format(len(content_dataset)))

    style = Image.open(opts.style_path).convert('RGB')
    Transforms = transforms.Compose([
        transforms.Resize((opts.style_size, opts.style_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    style = Transforms(style)
    style = style.unsqueeze(0)
    style = style.repeat(opts.test_batch_size,1,1,1)
    global_i = 0


    for content in tqdm(content_dataloader):
        with torch.no_grad():
            x  = content.cuda().float()
            t,c,h,w = x.size()
            x = x.view(1,t,c,h,w)
            y  = style.cuda().float()
            result_batch = run_on_batch( x , y , net)


        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            im_path = content_dataset.conent_paths[global_i]
            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)
            global_i += 1



def run_on_batch(x , y , net):
    result_batch = net(x,y,tab = 'inference')
    return result_batch


if __name__ == '__main__':
    run()
