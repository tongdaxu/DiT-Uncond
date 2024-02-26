import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS, SASolverSampler
from tools.download import find_model
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2, DiT_XL_2
from diffusion.data.datasets import get_chunks, ASPECT_RATIO_256_TEST, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST
from data.dataloader import get_dataset, get_dataloader
import yaml
import torchvision.transforms as transforms
from condition_methods import get_conditioning_method
from measurements import get_noise, get_operator

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--tokenizer_path', default='/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins_share/sd-vae-ft-ema', type=str)
    parser.add_argument('--n', default=1000, type=int)
    parser.add_argument('--model_path', default='/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins/PixArt-XL-2-SAM-256x256.pth', type=str)
    # parser.add_argument('--model_path', default='/NEW_EDS/JJ_Group/xutd/PixArt-alpha/output/trained_model/checkpoints/epoch_4_step_10000.pth', type=str)
    parser.add_argument('--bs', default=8, type=int)
    parser.add_argument('--cfg_scale', default=0.0, type=float)
    parser.add_argument('--sampling_algo', default='iddpm', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=1000, type=int)
    parser.add_argument('--save_name', default='test_sample', type=str)
    parser.add_argument('--model_type', default='pixart', type=str)

    return parser.parse_args()


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(True)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)


def visualize(items, bs, sample_steps, cfg_scale):
    task_config = load_yaml("/NEW_EDS/JJ_Group/xutd/PixArt-alpha/super_resolution.yaml")    
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):
    for i, ref_img in enumerate(loader):
        ref_img = ref_img.to('cuda')
        hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
        ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
        latent_size_h, latent_size_w = latent_size, latent_size
        null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]
        
        y = operator.forward(ref_img)
        
        caption_embs, emb_masks = null_y, None
        n = 1
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
        model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                            cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        diffusion = IDDPM(str(sample_steps))
        
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, measurement=y, measurement_cond_fn=measurement_cond_fn, vae=vae, noise=z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        samples.unsqueeze_(0)
        samples = vae.decode(samples / 0.18215).sample
        torch.cuda.empty_cache()
        
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        save_image(samples, "sample256_new2_"+str(i)+".png", nrow=1, normalize=True, value_range=(-1, 1))

if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    lewei_scale = {256: 1, 512: 1, 1024: 2}     # trick for positional embedding interpolation
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.float16
    print(f"Inference with {weight_dtype}")

    model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)

    print(f"Generating sample from ckpt: {args.model_path}")
    state_dict = find_model(args.model_path)
    if 'state_dict' not in state_dict:
        state_dict['state_dict'] = state_dict
    del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)

    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)

    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)
    work_dir = os.path.join(*args.model_path.split('/')[:-2])
    work_dir = f'/{work_dir}' if args.model_path[0] == '/' else work_dir

    items = ['{0:05d}'.format(i) for i in range(args.n)]

    # img save setting
    try:
        epoch_name = re.search(r'.*epoch_(\d+).*.pth', args.model_path).group(1)
        step_name = re.search(r'.*step_(\d+).*.pth', args.model_path).group(1)
    except Exception:
        epoch_name = 'unknown'
        step_name = 'unknown'
    img_save_dir = os.path.join(work_dir, 'vis')
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_{args.dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}_seed{seed}")
    os.makedirs(save_root, exist_ok=True)
    visualize(items, args.bs, sample_steps, args.cfg_scale)