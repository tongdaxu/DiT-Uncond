_base_ = ['/NEW_EDS/JJ_Group/xutd/PixArt-alpha/configs/PixArt_xl2_internal.py']
data_root = '/NEW_EDS/JJ_Group/xutd/PixArt-alpha'

data = dict(type='UncondDataset', root='/NEW_EDS/JJ_Group/xutd/PixArt-alpha/tools/imagenet_256_sd_list.txt', dummy_caption_path='/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins_share/dummy_caption.npz', transform='default_train')
image_size = 256

# model setting
window_block_indexes = []
window_size=0
use_rel_pos=False
model = 'PixArt_XL_2'
fp32_attention = True
load_from = "/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins_share/PixArt-XL-2-SAM-256x256.pth"
vae_pretrained = "/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins_share/sd-vae-ft-ema"
lewei_scale = 1.0

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=10
train_batch_size = 64 # 16 for 512, 64 for 256
num_epochs = 200 # 3
gradient_accumulation_steps = 2
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 200
log_interval = 20
save_model_steps=100
work_dir = 'output/debug'
class_dropout_prob = 1.0