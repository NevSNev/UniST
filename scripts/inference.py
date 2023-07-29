import os
from argparse import Namespace
from tqdm import tqdm
from tqdm.contrib import tzip
import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import DataLoader
import sys
sys.path.append(".")
sys.path.append("..")
from datasets.images_dataset import ImagesDataset
from datasets.style_dataset import StyleDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.model import video_Style_transfer
import torchvision.transforms as transforms


def run():
    test_opts = TestOptions().parse()

    out_path_results_image = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_results_video = os.path.join(out_path_results_image, 'video')
    os.makedirs(out_path_results_image, exist_ok=True)
    os.makedirs(out_path_results_video, exist_ok=True)

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
    dataset_content = ImagesDataset(image_root=opts.image_path,video_root=opts.video_path,
                                    content_transform=transforms_content,
                                    opts=opts)

    dataset_style = StyleDataset(style_root=opts.style_image_path,
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

    style_video = Image.open(opts.style_video_path).convert('RGB')
    Transforms = transforms.Compose([
        transforms.Resize((opts.style_size, opts.style_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    style_video = Transforms(style_video)
    style_video = style_video.unsqueeze(0).repeat(opts.test_batch_size,1,1,1)


    for  (image,video),style in tqdm(tzip(content_dataloader,style_dataloader)):
        with torch.no_grad():
            image = image.cuda().float()
            t,c,h,w = image.size()
            video = video.cuda().float()
            content = torch.cat([image,video],dim=0)
            content = content.view(1,-1,c,h,w)
            style = style.cuda().float()
            style_video = style_video.cuda().float()
            style = torch.cat([style,style_video],dim=0)
            result_batch = run_on_batch(content, style, net, opts)


        for i in range(opts.test_batch_size*2):
            if i < opts.test_batch_size:
                result = tensor2im(result_batch[i])
                im_path = dataset_content.image_path[global_i]
                im_save_path = os.path.join(out_path_results_image, os.path.basename(im_path))
                print(im_save_path)
                Image.fromarray(np.array(result)).save(im_save_path)
                global_i += 1
                if i == 1:
                    global_i -=2
            else :
                result = tensor2im(result_batch[i])
                im_path = dataset_content.video_path[global_i]
                im_save_path = os.path.join(out_path_results_video, os.path.basename(im_path))
                print(im_save_path)
                Image.fromarray(np.array(result)).save(im_save_path)
                global_i += 1


def run_on_batch(content,style, net, opts):
    y_hat = net.forward(content,style, tab = 'inference')
    return y_hat


if __name__ == '__main__':
    run()
