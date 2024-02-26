## Unconditional Finetune of
* PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha
* DiT: https://github.com/facebookresearch/DiT

## Dataset
* Any folder contains images, with subfolder or not, may be used. You do not need to resize those images, the script will handle this.
* We need to extra VAE feature from those images, to do so, we need
  * First, download the VAE model. The VAE model typically used for is the stable diffusion vae
    * stabilityai/sd-vae-ft-ema: https://huggingface.co/stabilityai/sd-vae-ft-ema
    * stabilityai/sd-vae-ft-mse: https://huggingface.co/stabilityai/sd-vae-ft-mse
    * The difference between those two vaes are minimal, so use either is ok.
  * Next, run the script to extra features using vaes
    ```bash
    python tools/extract_features_uncond.py --img_size $IMAGE_SIZE --pretrained_models_dir $VAE_DIR --dataset_root $IMAGE_DATASET_DIR --dataset_list $IMAGE_DATASET_SUB_DIR --vae_save_root $OUT_PUT_DIR
    ```
    * Typically the IMAGE_DATASET_DIR and IMAGE_DATASET_SUB_DIR can be the same location. However, sometimes you might want to process different subfolders of a dataset in parallel. For example, you have a ImageNet dataset
      ```bash
      --train
      ----n15075141
      ----n13133613
      ----...
      ```
    * And you want to process n15075141 and n13133613 subfolders in parallel, then the best way is to run two commands in parallel
      ```bash
      CUDA_VISIBLE_DEVICES=0 python tools/extract_features_uncond.py --img_size $IMAGE_SIZE --pretrained_models_dir $VAE_DIR --dataset_root train/ --dataset_list train/n15075141 --vae_save_root $OUT_PUT_DIR

      CUDA_VISIBLE_DEVICES=1 python tools/extract_features_uncond.py --img_size $IMAGE_SIZE --pretrained_models_dir $VAE_DIR --dataset_root train/ --dataset_list train/n13133613 --vae_save_root $OUT_PUT_DIR
      ```
    * And this allows a structured output
      ```bash
      --OUT_PUT_DIR
      ----n15075141
      ----n13133613
      ----...
      ```
  * (optional) list the data
    * Typically it is not efficient to feed a whole folder to dataloader and let it glob or discover files. This brings unnecessary io as this happens everytime training begins. We recommend to do it once and save a list, and pass that list to root directly: 
      ```bash
      python tools/folder_to_txt.py
      ```

## Finetune an existing model
* From PixArt Alpha 256x256
  ```bash
  python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 train.py \
  ./notebooks/PixArt_xl2_img256_uncond.py \
  --work-dir output/trained_256x256_imagenet_pa \
  --loss_report_name="train_loss"
  ```
* From DiT 256x256
  ```bash
  python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 train.py \
  ./notebooks/DiT_XL_2_img256_uncond.py \
  --work-dir output/trained_256x256_imagenet_dit \
  --loss_report_name="train_loss"
  ```

## Note:
* This repo is a fork based on https://github.com/PixArt-alpha/PixArt-alpha
* The original Readme: https://github.com/PixArt-alpha/PixArt-alpha/blob/master/README.md
