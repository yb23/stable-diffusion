import torch
import numpy as np
import PIL
from PIL import Image, ImageDraw
import random
import cv2
import wandb
from inpainting_pipeline import StableDiffusionInpaintPipeline  

PROMPTS_START = ["aerial view of ", "top view of ", "satellite view of "]
CLASSES_NAMES = np.array(["","building","pervious surface","road","bare soil","water","coniferous trees","deciduous trees","brushwood","vineyard","grass","agricultural vegetation","plowed land","swimming pool","snow","cut","mixed","lignous","greenhouse",""])
FREQ_CLASSES = np.array([0., 8.14, 8.25, 13.72, 3.47, 4.88, 2.74, 15.38, 6.95, 3.13, 17.84, 10.98, 3.88, 0., 0., 0., 0., 0., 0., 0.])
FREQ_CLASSES = FREQ_CLASSES / FREQ_CLASSES.sum()

def prepare_mask_and_masked_image(image, mask, toTensor=True, mask_before_norm=False):  
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    if toTensor:
        mask = torch.from_numpy(mask)
    if toTensor and mask_before_norm:
        image = torch.from_numpy(image).to(dtype=torch.float32)
        masked_image = (image * (mask < 0.5)) / 127.5 - 1.0
        image = image / 127.5 - 1.0
    elif toTensor:
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        masked_image = image * (mask < 0.5)
    else:
        masked_image = image * (mask < 0.5)

    return mask, masked_image


# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False, min_size=0):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    if (min_size==0) or (min_size>=int(im_shape[0] * ratio)):
        size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    else:
        size = (random.randint(min_size, int(im_shape[0] * ratio)), random.randint(min_size, int(im_shape[1] * ratio)))
    if mask_full_image:
        draw.rectangle((0, 0, im_shape[0], im_shape[1]), fill=255,)
        return mask
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle((center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),fill=255,)
    else:
        draw.ellipse((center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),fill=255,)
    return mask


def getMaskedObjects(labels,mask):
    freq = np.bincount((labels[0] * mask.astype(int)).flatten(), minlength=20) / mask.sum()
    freq[0] = 0
    classes = CLASSES_NAMES[(np.nan_to_num(freq/FREQ_CLASSES)>1)]
    return " and ".join(classes)


def log_validation(logger, val_images, val_img_mask, val_prompts, val_random_prompts, vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    infrarouges = []
    altis = []
    prompts = []
    prob_blur = 0.8 if args.mask_blur else 0.2
    for i in range(len(val_img_mask)):
        if np.random.randint(1,10)>=int((1-prob_blur)*10):
            ker = np.random.randint(1,100)*2-1
            q = np.random.randint(1,40)
            mask = cv2.GaussianBlur(val_img_mask[i][0][0][0].astype(float), (ker,ker), q).astype(float)
        else:
            mask = val_img_mask[i][0][0][0].astype(float)
        img = np.moveaxis(val_images[i],0,2)
        img_mask = np.moveaxis(val_img_mask[i][1][0],0,2)
        img_orig = np.moveaxis(val_images[i],0,2)
        prompt = PROMPTS_START[np.random.randint(0,2)] + val_prompts[i] + ", high resolution, highly detailed"
        if args.cpu:
            image = pipeline(prompt=[prompt], num_inference_steps=20, generator=generator, image=[img], mask_image=[mask], output_type="np").images[0]
        else:
            with torch.autocast("cuda"):
                image = pipeline(prompt=[prompt], num_inference_steps=20, generator=generator, image=[img], mask_image=[mask], output_type="np").images[0]
        
        grid = np.concatenate([np.concatenate((img_orig[:,:,:3],img_mask[:,:,:3]), axis=1),
                                (np.concatenate((image[:,:,:3], np.tile(mask[:,:,None],(1,1,3))),axis=1)*255).astype(int)], axis=0)
        gridIR = np.concatenate((img_orig[:,:,3],(image[:,:,3]*255).astype(int)), axis=0)
        gridAlti = np.concatenate((img_orig[:,:,4],(image[:,:,4]*255).astype(int)), axis=0)
                                
        images.append(grid)
        infrarouges.append(gridIR)
        altis.append(gridAlti)
        prompts.append(prompt)
    
    ch, H,W = val_images[0].shape
    mask = np.ones([H,W])
    img_orig = np.zeros([H,W,ch])
    random_prompt = PROMPTS_START[np.random.randint(0,2)] + val_random_prompts[random.randint(0, len(val_random_prompts)-1)] + ", high resolution, highly detailed"
    if args.cpu:
        image = pipeline(prompt=[random_prompt], num_inference_steps=20, generator=generator, image=[img_orig], mask_image=[mask.astype(float)], output_type="np").images[0]
    else:
        with torch.autocast("cuda"):
            image = pipeline(prompt=[random_prompt], num_inference_steps=20, generator=generator, image=[img_orig], mask_image=[mask.astype(float)], output_type="np").images[0]
    
    grid = np.concatenate((img_orig[:,:,:3],(image[:,:,:3]*255).astype(int)), axis=0)
    gridIR = np.concatenate((img_orig[:,:,3],(image[:,:,3]*255).astype(int)), axis=0)
    gridAlti = np.concatenate((img_orig[:,:,4],(image[:,:,4]*255).astype(int)), axis=0)
                            
    images.append(grid)
    infrarouges.append(gridIR)
    altis.append(gridAlti)
    prompts.append(random_prompt)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log({"validation": [wandb.Image(image, caption=f"{i}: {prompts[i]}") for i, image in enumerate(images)]})
            tracker.log({"infrarouge": [wandb.Image(image, caption=f"{i}: {prompts[i]}") for i, image in enumerate(infrarouges)]})
            tracker.log({"altitude": [wandb.Image(image, caption=f"{i}: {prompts[i]}") for i, image in enumerate(altis)]})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images

def prepare_mask_and_masked_image_for_pipeline(image, mask, height, width, return_image: bool = False):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. 
    *** (REMOVED) The ``mask`` will be binarized (``mask > 0.5``) and cast to ``torch.float32`` too. -> NO BINARIZATION OF THE MASK ***

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        ################"  No binarization of the mask   ##################################"
        #mask[mask < 0.5] = 0
        #mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)


        #mask[mask < 0.5] = 0
        #mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    ########################################################
    masked_image = image * (1-mask) #(mask < 0.5)
    ########################################################
    
    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image