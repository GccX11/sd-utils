# utility functions to make it easier to work with the diffusion model
# based on this notebook:
# https://colab.research.google.com/drive/1_kbRZPTjnFgViPrmGcUsaszEdYa8XTpq?usp=sharing

# write out interpolation video
# get latents for image
# perturb latents for a given image to generate new images


from base64 import b64encode

import cv2
import os
import numpy as np
from IPython.display import HTML


import PIL
import torch
from torch import autocast
from tqdm.auto import tqdm
from PIL import Image, ImageDraw
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


def init():
    # tell the method these should be global variables
    # this is helpful for flywheeling, and keeping the big
    # stuff in memory while we play with it
    global device
    global pipe
    global vae
    global tokenizer
    global text_encoder
    global unet
    global scheduler
    global im2im_scheduler
    
    device = 'cuda'

    # make sure you're logged in with 'huggingface-cli login'
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', revision='fp16',
        torch_dtype=torch.float16, use_auth_token=True)
    pipe = pipe.to(device)
    
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=True)
    vae = vae.to(device)
    
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = text_encoder.to(device)
    
    # 3. The UNet model for generating the latents
    unet = UNet2DConditionModel.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='unet', use_auth_token=True) \
        .to(device)
    
    # 4. Create a scheduler for inference
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000)
    
    # New scheduler for img-to-img
    im2im_scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000)
    

def decode_img_latents(latents):
    """
    decode_img_latents "converts" latents to list of imgs

    :param latents: list of latents to be decoded
    :return: the list of imgs decoded from the latents
    """

    global vae
    
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        imgs = vae.decode(latents)['sample']

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs]
    return pil_images


def encode_img_latents(imgs):
    """
    encode_img_latents "converts" the imgs to a list of latents

    :param imgs: list of imgs to be encoded
    :return: list of latents encoded by the images
    """
    
    global vae

    if not isinstance(imgs, list):
        imgs = [imgs]

    img_arr = np.stack([np.array(img) for img in imgs], axis=0)
    img_arr = img_arr / 255.0
    img_arr = torch.from_numpy(img_arr).float().permute(0, 3, 1, 2)
    img_arr = 2 * (img_arr - 0.5)

    latent_dists = vae.encode(img_arr.to(device))
    latent_samples = latent_dists.latent_dist.sample()
    latent_samples *= 0.18215

    return latent_samples


def get_text_embeds(prompt):
    """
    "converts" the text prompt to text embeddings
    TODO: figure out what these embeddings are
    
    :param prompt: single text prompt
    :return: text embeddings
    """
    
    global tokenizer
    global text_encoder
    
    # Tokenize text and get embeddings
    text_input = tokenizer(
        prompt, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt')
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    return text_embeddings


def imgs_to_video(imgs, video_name='video.mp4', fps=15):
    """
    imgs_to_video saves the stack of images to a video
    
    :param imgs: stack of images to save
    :param filename: name of the video file (including extension)
    :param fps: frames per second (i.e. speed) of the video
    """
    # Source: https://stackoverflow.com/questions/52414148/turn-pil-images-into-video-on-linux
    video_dims = (imgs[0].width, imgs[0].height)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, fourcc, fps, video_dims)
    for img in imgs:
        tmp_img = img.copy()
        video.write(cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR))
    video.release()


def perturb_latents(latents, scale=0.1):
    """
    perturb_latents "moves" the latents arount by the specified amount
    TODO: need to figure out what this is actually doing
    
    :param latents: latents to perturb
    :param scale: amount to perturb the image (scale)
    :return: the perturbed latents
    """
    noise = torch.randn_like(latents)
    new_latents = (1 - scale) * latents + scale * noise
    return (new_latents - new_latents.mean()) / new_latents.std()


def image_grid(imgs, rows, cols):
    """
    draws the images in a grid
    
    :param imgs: list of images to draw
    :param rows: # rows or height
    :param cols: # cols or width
    """
    assert len(imgs) == rows*cols

    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def produce_latents(text_embeddings, height=512, width=512,
                    num_inference_steps=50, guidance_scale=7.5, latents=None,
                    return_all_latents=False, return_start_latent=False,
                    start_latent=None, start_step=10):
    """
    produce_latents 
    
    :param text_embeddings: 
    :param height: 
    :param width: 
    :param num_inference_steps: 
    :param guidance_scale: 
    :param latents: 
    :param return_all_latents:
    :param start_step: 
    :param noise: noise to use when encoding the image
    :return: 
    """
    
    global scheduler
    global unet

    if start_latent is not None:
        latents = start_latent
    else:
        start_latent = torch.randn((text_embeddings.shape[0], unet.in_channels, \
                                    height // 8, width // 8))
        latents = start_latent
    
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0] # I think I need this?

    latents = latents.to(device)

    if start_step > 0:
        start_timestep = scheduler.timesteps[start_step]
        start_timesteps = start_timestep.repeat(latents.shape[0]).long()

        noise = torch.randn_like(latents)
        latents = scheduler.add_noise(latents, noise, start_timesteps)
    
    latent_hist = [latents]
    with autocast('cuda'):
        for i, t in tqdm(enumerate(scheduler.timesteps[start_step:])):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = latents
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
            
            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents)['prev_sample']
            
            latent_hist.append(latents)

    if return_start_latent:
        return latents, start_latent

    if not return_all_latents:
        return latents

    all_latents = torch.cat(latent_hist, dim=0)
    return all_latents


def produce_latents_video(text_embeddings, height=512, width=512,
                     num_inference_steps=50, guidance_scale=7.5, latents=None,
                     return_all_latents=False, start_step=10):
    """
    produce_latents_video 
    
    :param text_embeddings: 
    :param height: 
    :param width: 
    :param num_inference_steps: 
    :param guidance_scale: 
    :param latents: 
    :param return_all_latents:
    :param start_step: 
    :return: 
    """
    
    global im2im_scheduler
    global unet

    assert latents is not None, "latents must be provided for this to be interesting"

    im2im_scheduler.set_timesteps(num_inference_steps)

    latents = latents.to(device)

    if start_step > 0:
        start_timestep = im2im_scheduler.timesteps[start_step]
        start_timesteps = start_timestep.repeat(latents.shape[0]).long()

        noise = torch.randn_like(latents)
        latents = im2im_scheduler.add_noise(latents, noise, start_timesteps)

    latent_hist = [latents]
    with autocast('cuda'):
        for i, t in tqdm(enumerate(im2im_scheduler.timesteps[start_step:])):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = im2im_scheduler.step(noise_pred, t, latents)['prev_sample']

            latent_hist.append(latents)

    if not return_all_latents:
        return latents

    all_latents = torch.cat(latent_hist, dim=0)
    return all_latents


def prompt_to_video(prompts, height=512, width=512, num_inference_steps=50,
                   guidance_scale=7.5, latents=None, return_all_latents=False,
                   batch_size=2, start_step=0):
    """
    prompt_to_video 
    
    :param prompts: 
    :param height: 
    :param width: 
    :param num_inference_steps: 
    :param guidance_scale: 
    :param latents: 
    :param return_all_latents:
    :param batch_size: 
    :param start_step: 
    :return: 
    """
    
    if isinstance(prompts, str):
        prompts = [prompts]

    # Prompts -> text embeds
    text_embeds = get_text_embeds(prompts)

    # Text embeds -> img latents
    latents = produce_latents_video(
        text_embeds, height=height, width=width, latents=latents,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        return_all_latents=return_all_latents, start_step=start_step)

    # Img latents -> imgs
    all_imgs = []
    for i in tqdm(range(0, len(latents), batch_size)):
        imgs = decode_img_latents(latents[i:i+batch_size])
        all_imgs.extend(imgs)

    return all_imgs



def embed_to_img(text_embeds, num_steps=50, start_latent=None):
    # Text embeds -> img latents
    if start_latent is None:
        latents, start_latent = produce_latents(
            text_embeds, height=512, width=512,
            num_inference_steps=num_steps, guidance_scale=7.5,
            return_all_latents=False, start_step=0,
            return_start_latent=True)
    else:
        latents = produce_latents(
            text_embeds, height=512, width=512,
            num_inference_steps=num_steps, guidance_scale=7.5,
            return_all_latents=False, start_step=0,
            return_start_latent=False, start_latent=start_latent)

    # Img latents -> imgs
    all_imgs = []
    for i in tqdm(range(0, len(latents), 2)):
        imgs = decode_img_latents(latents[i:i+2])
        all_imgs.extend(imgs)

    return all_imgs, start_latent


def prompt_to_latent(prompts, height=512, width=512, num_inference_steps=10,
                  guidance_scale=7.5, latents=None, return_all_latents=False,
                  batch_size=2, start_step=0):
    """
    prompt_to_latent
    
    :param prompts: 
    :param height: 
    :param width: 
    :param num_inference_steps: 
    :param guidance_scale: 
    :param latents: 
    :param return_all_latents:
    :param batch_size: 
    :param start_step: 
    :return: 
    """

    if isinstance(prompts, str):
        prompts = [prompts]

    # Prompts -> text embeds
    text_embeds = get_text_embeds(prompts)

    # Text embeds -> img latents
    latents = produce_latents(
        text_embeds, height=height, width=width, latents=latents,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        return_all_latents=return_all_latents, start_step=start_step)

    return latents


def prompt_to_img(prompts, height=512, width=512, num_inference_steps=10,
                  guidance_scale=7.5, latents=None, return_all_latents=False,
                  batch_size=2, start_step=0):
    """
    prompt_to_img 
    
    :param prompts: 
    :param height: 
    :param width: 
    :param num_inference_steps: 
    :param guidance_scale: 
    :param latents: 
    :param return_all_latents:
    :param batch_size: 
    :param start_step: 
    :return: 
    """
    
    if isinstance(prompts, str):
        prompts = [prompts]

    # Prompts -> text embeds
    text_embeds = get_text_embeds(prompts)

    # Text embeds -> img latents
    latents = produce_latents(
        text_embeds, height=height, width=width, latents=latents,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
        return_all_latents=return_all_latents, start_step=start_step)

    # Img latents -> imgs
    all_imgs = []
    for i in tqdm(range(0, len(latents), batch_size)):
        imgs = decode_img_latents(latents[i:i+batch_size])
        all_imgs.extend(imgs)

    return all_imgs


#################################### API ####################################


def im(text, start=0, end=10, h=512, w=512, prior=None, interpolate=False):
    """
    im generates images based on text
    T -> Z -> X
    
    :param text
    :param start
    :param end
    :param h
    :param w
    :param prior
    :param interpolate
    :return: list of images
    """
    return prompt_to_img(text, height=h, width=w, start_step=start,
                          num_inference_steps=end, latents=prior,
                          return_all_latents=interpolate)

# generate image based on text
# find latents best matching the provided text
# T -> Z -> Xp
def video(text, start=0, end=50, h=512, w=512, prior=None, interpolate=True):
    """
    video generates stack of images based on text prompt
    TODO: does this return a list of lists of images?
          or just one list of images for the prompt?
    
    :param text
    :param start
    :param end
    :param h
    :param w
    :param prior
    :param interpolate
    :return: stack (list) of images
    """
    return prompt_to_video(text, height=h, width=w, start_step=start,
                          num_inference_steps=end, latents=prior,
                          return_all_latents=interpolate)


def encode(im):
    """
    encode "converts" the imgs to a list of latents
    X -> Z

    :param imgs: list of imgs to be encoded
    :return: list of latents encoded by the images
    """
    return encode_img_latents([im])


def decode(latents):
    """
    decode "converts" latents to list of imgs
    Z -> X
    
    :param latents: list of latents to be decoded
    :return: the list of imgs decoded from the latents
    """
    return decode_img_latents(latents)


def perturb(latents, amount):
    """
    perturb "moves" the latents arount by the specified amount
    TODO: need to figure out what this is actually doing
    
    :param latents: latents to perturb
    :param amount: amount to perturb the image (scale)
    :return: the perturbed latents
    """
    return perturb_latents(latents, scale=amount)


def save_video(imgs, filename, fps=15):
    """
    save_video saves the stack of images to a video
    
    :param imgs: stack of images to save
    :param filename: name of the video file (including extension)
    :param fps: frames per second (i.e. speed) of the video
    """
    imgs_to_video(imgs, video_name=filename, fps=fps)

