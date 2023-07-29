import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from utils import data_utils
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.utils import Stack, ToTorchFormatTensor


class Video_image_Dataset(torch.utils.data.Dataset):
    def __init__(self, video_data_root,image_data_root,image_transform):
        self.video_data_root = video_data_root
        self.image_data_root = sorted(data_utils.make_dataset(image_data_root))
        self.w = 256
        self.h = 256
        self.sample_length = 2
        self.size = (self.w, self.h)

        video_list=os.listdir(self.video_data_root)
        video_list.sort()
        self.video_names = video_list

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

        self.image_transform = image_transform

    def __len__(self):

        return min(len(self.video_names),len(self.image_data_root))

    def __getitem__(self, index):

        imgs = self.load_item_image(index)
        frames = self.load_item_video(index)
        input = torch.cat([imgs,frames],dim=0)
        return input


    def load_item_video(self, index):
        video_name = self.video_names[index]
        image_names = os.listdir(os.path.join(self.video_data_root+'/'+video_name))
        # [f"{str(i).zfill(5)}.jpg" for i in range(1,len(image_names))]
        all_frames =[f"frame_{str(i).zfill(4)}.png" for i in range(1,len(image_names)+1)]
        ref_index = get_ref_index_video(len(all_frames), self.sample_length)
        frames = []
        for idx in ref_index:
            img = Image.open(os.path.join(self.video_data_root+'/'+video_name+'/'+all_frames[idx])).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)

        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        return frame_tensors


    def load_item_image(self, index):
        ref_index = get_ref_index_image((len(self.image_data_root)), self.sample_length)
        images = []
        for idx in ref_index:
            img = Image.open(self.image_data_root[idx]).convert('RGB')
            img = self.image_transform(img)
            images.append(img)
        images_tensors = torch.stack(images,dim=0)
        return images_tensors

def get_ref_index_video(length, sample_length):
    pivot = random.randint(0, length-sample_length)
    ref_index = [pivot+i for i in range(sample_length)]

    return ref_index


def get_ref_index_image(length, sample_length):
    ref_index = []
    for i in range(sample_length):
        pivot = random.randint(0, length-1)
        ref_index.append(pivot)
    return ref_index