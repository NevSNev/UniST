import os
import numpy as np
import torch
import sys
sys.path.append(".")
sys.path.append("..")
from PIL import Image
from PIL import ImageFile
from argparse import Namespace
from tqdm import tqdm
from tqdm.contrib import tzip
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import DataLoader
from datasets.images_dataset import ImagesDataset
from datasets.style_dataset import StyleDataset
from utils.common import tensor2im
from options.test_options import TestOptions
from models.model import video_Style_transfer
import torchvision.transforms as transforms


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

    transforms_content = transforms.Compose([
        transforms.Resize((opts.content_size, opts.content_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transforms_style = transforms.Compose([
        transforms.Resize((opts.style_size, opts.style_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_content = ImagesDataset(content_root=opts.content_path,
                                    content_transform=transforms_content,
                                    opts=opts)

    dataset_style = StyleDataset(style_root=opts.style_path,
                                 style_transform=transforms_style,
                                 opts=opts)

    content_dataloader = DataLoader(dataset_content,
                                    batch_size=opts.test_batch_size,
                                    shuffle=False,
                                    num_workers=int(opts.test_workers),
                                    drop_last=True)
    style_dataloader = DataLoader(dataset_style,
                                  batch_size=opts.test_batch_size,
                                  shuffle=False,
                                  num_workers=int(opts.test_workers),
                                  drop_last=True)


    global_i = 0

    for  content,style in tqdm(tzip(content_dataloader,style_dataloader)):
        with torch.no_grad():
            content = content.cuda().float()
            t,c,h,w = content.size()
            content = content.view(1,t,c,h,w)
            style = style.cuda().float()
            result_batch = run_on_batch(content,style, net, opts)


        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            im_path = dataset_content.conent_paths[global_i]
            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)
            global_i += 1



def run_on_batch(content,style, net, opts):
    y_hat = net.forward(content,style,tab = 'inference')
    return y_hat


if __name__ == '__main__':
    run()
