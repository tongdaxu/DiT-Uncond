import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import json
from tqdm import tqdm
import argparse
import threading
from queue import Queue
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.folder import default_loader
from glob import glob

from diffusers.models import AutoencoderKL


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, droot, transform=None):
        self.droot = droot
        self.images = []
        cnt = 0

        if self.droot[-3:] == 'txt':
            f = open(self.droot, 'r')
            lines = f.readlines()            
            for line in lines:
                self.images.append(line.strip())
        else:
            for root, dirs, files in os.walk(self.droot):
                for file in files:
                    if not ".jpg" in file:
                        continue
                    line = os.path.join(root, file)
                    self.images.append(line)
                    cnt += 1
                    if cnt % 1000 == 0:
                        print("discover {} files".format(cnt))

        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.transform is not None:
            return self.transform(img), self.images[index]
        else:
            return img, self.images[index]

    def __len__(self):
        return len(self.images)

def extract_img_vae_batch():
    vae = AutoencoderKL.from_pretrained(f'{args.pretrained_models_dir}/sd-vae-ft-ema').to(device)
    lines = []
    cnt = 0
    vae_save_root = f'{args.vae_save_root}/{image_resize}resolution'
    os.umask(0o000)
    os.makedirs(vae_save_root, exist_ok=True)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(image_resize, InterpolationMode.LANCZOS),  # Image.BICUBIC
        T.CenterCrop(image_resize),
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ])
    dataset = ImageDataset(args.dataset_list, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=64, pin_memory=True, persistent_workers=False)

    os.umask(0o000)  # file permission: 666; dir permission: 777
    for i, (imgs, lines) in tqdm(enumerate(dataloader)):
        imgs = imgs.to(device)
        with torch.no_grad():
            posterior = vae.encode(imgs).latent_dist
            z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy()
        for b in range(imgs.shape[0]):
            save_path = os.path.join(vae_save_root, os.path.relpath(lines[b], args.dataset_root).split('.')[0])
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, z[b])

def get_args():
    # 1 gpu 20 days
    # 45 5 gpu, 4 gpu -> 1 week
    # 5 queue -> id % 5 
    # for [0,999] xxx:
    #     if in hist:
    #           continue
    #     poll 
    # subprocess.call(CUDA_VISIBLE_DEVICE = n % 5 python extrat_features_uncond)

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default=512, type=int, help="image scale for multi-scale feature extraction")
    parser.add_argument('--pretrained_models_dir', default='/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins_share', type=str)
    # not real root
    # fix
    # /NEW_EDS/JJ_Group/xutd/common_datasets/sam_unzip
    parser.add_argument('--dataset_root', default='/NEW_EDS/JJ_Group/xutd/common_datasets/sam_unzip', type=str)
    # real root
    # change
    # /NEW_EDS/JJ_Group/xutd/common_datasets/sam_unzip/sa_000988
    parser.add_argument('--dataset_list', default='/NEW_EDS/JJ_Group/xutd/common_datasets/sam_unzip/sa_000029', type=str)
    # save folder
    # fix, subfolder = dataset_list - dataset_root
    parser.add_argument('--vae_save_root', default='/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins_share/sam_sd_vae', type=str)
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_resize = args.img_size

    # prepare extracted image vae features for training
    print(f'Extracting Single Image Resolution {image_resize}')
    # extract_img_vae()
    extract_img_vae_batch()