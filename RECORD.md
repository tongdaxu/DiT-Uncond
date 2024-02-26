CUDA_VISIBLE_DEVICES=7 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python nohup python -u ./tools/extract_features.py \
--img_size 256 \
--json_path "/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins/data_info_256train_7.json" \
--t5_save_root "/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins/caption_feature_wmask" \
--vae_save_root "/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins/img_vae_features" \
--pretrained_models_dir "/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins" \
--dataset_root "" &> 7.out &

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python nohup python -u -m torch.distributed.launch --nproc_per_node=7 --master_port=12345 train.py \
./notebooks/PixArt_xl2_img512_internal_for_pokemon_sample_training.py \
--work-dir output/trained_model \
--loss_report_name="train_loss" &


python train.py ./notebooks/PixArt_xl2_img512_internal_for_pokemon_sample_training.py --work-dir output/trained_model --loss_report_name="train_loss"

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python 

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python inference.py

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 python inference.py

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 python inference.py

## Dataset
* SAM: 25M Photo
* ART: 25M Art

* 256x256 SAM training
* 256x256 SAM -> 256x256 Art  
* 256x256 Art -> 512x512 Art 
* 512x512 Art -> 1024x1024 Art 

## 256x256 
### No finetune SAM
* fid: 42.97
### No finetune DIT pre-train
* fid: 56.84
### No finetume ART
* fid: 65.36
### finetune SAM 2 ImageNet 1e4
* fid: 24.88
### finetune SAM 2 ImageNet 2e4
* fid: 24.65
### finetune SAM 2 ImageNet 3e4
* fid: 26.68
### finetune DIT 2 ImageNet 1e4
### finetune DIT 2 ImageNet 2e4
### finetune DIT 2 ImageNet 3e4

## 512x512 
### No finetune ART
* fid: 66.58
## No finetune DIT pre-train 
* fid: 71.60
## finetune ART 2 ImageNet 1e4

## finetune ART 2 ImageNet 2e4
* fid: 36.45

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=7 --master_port=12345 train.py \
./notebooks/PixArt_xl2_img512_internal_for_pokemon_sample_training.py \
--work-dir output/trained_model \
--loss_report_name="train_loss"


CUDA_VISIBLE_DEVICES=7 nohup python tools/extract_features.py \
--img_size 512 \
--json_path "/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins/data_info_512train_7.json" \
--t5_save_root "/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins/caption_feature_wmask" \
--vae_save_root "/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins/img_vae_features" \
--pretrained_models_dir "/NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins" \
--dataset_root "" &> 7.out &



nohup python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 train.py \
./notebooks/PixArt_xl2_img512_uncond.py \
--work-dir output/trained_model_tmp \
--loss_report_name="train_loss" &


https://scontent.xx.fbcdn.net/m1/v/t6/An_YmP5OIPXun-vu3hkckAZZ2s4lPYoVkiyvCcWiVY21mu1Ng5_1HeCa2CWiSTsskj8HQ8bN013HxNpYDdSC_7jWQq_svcg.tar?ccb=10-5&oh=00_AfBxIXg-7PKiUVqy4eRnWiiQVHLeLgX0P-BhXYRbPPNWZQ&oe=65EEB368&_nc_sid=0fdd51


python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 train.py \
./notebooks/PixArt_xl2_img256_uncond.py \
--work-dir output/trained_model_tmp \
--loss_report_name="train_loss"


CUDA_VISIBLE_DEVICES=1 python inference.py --model_type dit --model_path /NEW_EDS/JJ_Group/xutd/PixArt-alpha/bins_share/DiT-XL-2-256x256.pt

nohup python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 train.py \
./notebooks/DiT_XL_2_img256_uncond.py \
--work-dir output/trained_256x256_imagenet_dit \
--loss_report_name="train_loss" &


nohup python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 train.py \
./notebooks/PixArt_xl2_img256_uncond.py \
--work-dir output/trained_256x256_imagenet_pa \
--loss_report_name="train_loss" &