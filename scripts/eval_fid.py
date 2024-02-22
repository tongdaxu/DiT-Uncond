import torch
from torchvision import transforms
from fid import fid_pytorch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import os
from PIL import Image

ref_path = '/NEW_EDS/JJ_Group/xutd/common_datasets/imagenet_256x256/val1k'
dis_path = '/NEW_EDS/JJ_Group/zhuzr/DiT_copy/samples/DiT-XL-2-pretrained-size-256-vae-mse-cfg-1.0-seed-0'

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images = []
        if root[-3:] == 'txt':
            f = open(root, 'r')
            lines = f.readlines()            
            for line in lines:
                self.images.append(line.strip())
        else:
            self.images = sorted(os.listdir(self.root))
            self.images = [
                os.path.join(self.root, x) for x in self.images
            ]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        w, h = img.size

        if self.transform is not None:
            return self.transform(img), self.images[index]
        else:
            return img, self.images[index]

    def __len__(self):
        return len(self.images)

test_transforms = transforms.Compose(
    [transforms.ToTensor()]
)

ref_dataset = ImageDataset(root=ref_path, transform=test_transforms)
ref_dataloader = torch.utils.data.DataLoader(ref_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
ref_dataloader = list(ref_dataloader)
dis_dataset = ImageDataset(root=dis_path, transform=test_transforms)
dis_dataloader = torch.utils.data.DataLoader(dis_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
dis_dataloader = list(dis_dataloader)

fid_computer = fid_pytorch()

avgs = {
    "fid": []
}

with torch.no_grad():
    fid_computer.clear_pools()
    for i, ((x, _), (x_hat, _)) in tqdm(enumerate(zip(ref_dataloader, dis_dataloader))):

        x1 = x.cuda()
        x2 = x_hat.cuda()

        unfold = nn.Unfold(kernel_size=(64, 64),stride=(64, 64))
        x1_unfold = unfold(x1).reshape(1, 3, 64, 64, -1)
        x1_unfold = torch.permute(x1_unfold, (0, 4, 1, 2, 3)).reshape(-1, 3, 64, 64)
        x2_unfold = unfold(x2).reshape(1, 3, 64, 64, -1)
        x2_unfold = torch.permute(x2_unfold, (0, 4, 1, 2, 3)).reshape(-1, 3, 64, 64)

        fid_computer.add_ref_img(x1_unfold)
        fid_computer.add_dis_img(x2_unfold)

    avgs['fid'].append(fid_computer.summary_pools())
    print(avgs['fid'])