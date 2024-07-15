from utils import *
from pathlib import Path
import pandas as pd
import numpy as np
import glob
import rasterio
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import torch.nn.functional as F
from maskToColor import convert_to_color


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        image_channels=5,
        doOneHot=False,
        doOneHotSimple=False,
        num_classes=19,
        for_controlnet=False,
        args = None,
    ):  
        self.for_controlnet = for_controlnet
        self.doOneHot = doOneHot
        self.doOneHotSimple = doOneHotSimple
        self.num_classes = num_classes
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.image_channels = image_channels
        self.mask_blur = args.mask_blur
        self.instance_data_root = Path(instance_data_root)
        self.mask_before_norm = args.mask_before_norm
        self.prop_full_mask = args.prop_full_mask
        self.prop_empty_prompt = args.prop_empty_prompt
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        dfPrompts = pd.read_pickle(args.prompts_path).fillna("")
        dfPrompts.loc[0,"prompt_end"] = ""
        if args.flair:
            self.instance_images_path = glob.glob(instance_data_root+"/**/*.tif", recursive=True)##list(Path(instance_data_root).iterdir())
            self.instance_labels_path = [x.replace("aerial","labels").replace("IMG","MSK").replace("img","msk") for x in self.instance_images_path]
            self.image_ids = [x.split("/")[-1][:-4] for x in self.instance_images_path]
            mapping = pd.read_csv(args.mapping_path, index_col=0).set_index("img_id")["file"].to_dict()
            self.image_ids = [mapping.get(x,0) for x in self.image_ids]
        else:
            #prompts_path = "/gpfswork/rech/jrj/commun/FLAIR1/OCS_BigPrompts2.csv"
            self.instance_images_path = glob.glob(instance_data_root+"/**/*-RVBIE.tif.tif", recursive=True)
            if args.images_to_remove_path is not None:
                df = pd.read_csv(args.images_to_remove_path, index_col=0)
                toRemove = set(df["file"])
                print(f"Removing Flair Test Images ({len(toRemove)} images)")
                self.instance_images_path = [x for x in self.instance_images_path if x.split("/")[-1] not in toRemove]
                print(f"Remaining images : {len(self.images_path)}")
                print("Removing Gers 2016 Images")
                self.instance_images_path = [x for x in self.instance_images_path if x.split("/")[-1][:9]!="D032_2016"]
            self.instance_labels_path = [x.replace("img","msk").replace("-RVBIE.tif","-MSK_FLAIR19-LABEL.tif") for x in self.instance_images_path]
            self.image_ids = [x.split("/")[-1] for x in self.instance_images_path]
            #dfPrompts = pd.read_csv(prompts_path, index_col=0).set_index("file")    ####################"" prompts.csv
            #self.instance_prompt = list(dfPrompts.loc[self.image_ids]["BigPrompt"])
        self.instance_prompt = list(dfPrompts.loc[self.image_ids]["prompt_end"])


        self.num_instance_images = len(self.instance_images_path)
        print("Total training images : "+str(self.num_instance_images))
        #self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        idxs_val = np.random.choice(self.num_instance_images, 6)
        self.val_images = [rasterio.open(self.instance_images_path[i]).read()[:image_channels] for i in idxs_val]
        self.val_labels = [rasterio.open(self.instance_labels_path[i]).read() for i in idxs_val]
        #self.val_image_ids = [self.instance_images_path[i].split("/")[-1][:-4] for i in idxs_val]
        self.val_image_ids = [self.instance_images_path[i].split("/")[-1] for i in idxs_val]
        if args.flair:
            self.val_image_ids = [mapping.get(x[:-4],0) for x in self.val_image_ids]
        self.val_img_mask = [prepare_mask_and_masked_image(img, random_mask(img.shape[1:], 1, False, min_size=100), toTensor=False) for img in self.val_images]
        val_prompt_start = np.array([getMaskedObjects(self.val_labels[i], el[0]) + " " for i,el in enumerate(self.val_img_mask)])
        print(len(val_prompt_start))
        #self.val_prompts = ["satellite view, France" for img in self.val_images]
        #self.val_prompts = list(dfPrompts.loc[self.val_image_ids]["BigPrompt"])
        #self.val_random_prompts = list(dfPrompts.sample(30)["BigPrompt"])
        self.val_prompts = list(val_prompt_start + dfPrompts.loc[self.val_image_ids]["prompt_end"].values)
        print(len(self.val_prompts))

        val_rd_prompt_start = np.array([" and ".join(random.sample(list(CLASSES_NAMES[1:-1]), np.random.randint(1,4))) for k in range(30)])
        self.val_random_prompts = list(val_rd_prompt_start + (" " + dfPrompts.sample(30)["prompt_end"]).values)
        print(self.val_prompts)
        print(self.val_random_prompts)

        if for_controlnet:
            self.val_labels = [flair.convert_to_color(x[0]) for x in self.val_labels]
            self.val_random_labels = [convert_to_color(x[0]) for x in self.val_random_labels]
            if self.doOneHotSimple:
                self.val_OneHot = [torch.Tensor(rasterio.open(self.instance_labels_path[i]).read()[0]).to(torch.long) for i in idxs_val]
                self.val_random_OneHot = [torch.Tensor(rasterio.open(self.instance_labels_path[i]).read()[0]).to(torch.long) for i in idxs_random_prompts]
            elif self.doOneHot:
                self.val_OneHot = [torch.moveaxis(F.one_hot(torch.Tensor(rasterio.open(self.instance_labels_path[i]).read()[0]-1).to(torch.long), num_classes=self.num_classes), 2,0) for i in idxs_val]
                self.val_random_OneHot = [torch.moveaxis(F.one_hot(torch.Tensor(rasterio.open(self.instance_labels_path[i]).read()[0]-1).to(torch.long), num_classes=self.num_classes), 2,0) for i in idxs_random_prompts]
            else:
                self.val_OneHot = None
                self.val_random_OneHot = None


        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                #transforms.ToTensor(),    ########################################################################################################
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image_raster = rasterio.open(self.instance_images_path[index % self.num_instance_images]).read() #Image.open(self.instance_images_path[index % self.num_instance_images])
        labels = rasterio.open(self.instance_labels_path[index % self.num_instance_images]).read()

        if self.for_controlnet:
            if self.doOneHotSimple:
                example["instance_labels"] = torch.Tensor(labels[0]).to(torch.long)
            elif self.doOneHot:
                example["instance_labels"] = torch.moveaxis(F.one_hot(torch.Tensor(instance_label_raster[0]-1).to(torch.long), num_classes=self.num_classes),2,0)
            else:
                instance_label = np.moveaxis(convert_to_color(labels[0]),2,0)
                example["instance_labels"] = torch.Tensor(instance_label/255.0)  ###############################################

        
        if self.image_channels == 3:
            instance_image_raster = instance_image_raster[:3]
        #if not instance_image.mode == "RGB":
            #instance_image = instance_image.convert("RGB")
        instance_image = torch.Tensor(instance_image_raster)
        instance_image = self.image_transforms_resize_and_crop(instance_image)
        example["PIL_images"] = instance_image_raster
        example["instance_images"] = ((instance_image / 127.5) - 1) #####################  self.image_transforms(instance_image)
        
        full_mask = (self.prop_full_mask>0) and (random.random()<self.prop_full_mask)
        mask = random_mask(instance_image_raster.shape[1:], 1, full_mask)    # mask = random_mask(pil_image.size, 1, False) ##########################################
        # prepare mask and masked image
        mask, masked_image = prepare_mask_and_masked_image(instance_image_raster, mask, toTensor=True, mask_before_norm=self.mask_before_norm)
        prompt = getMaskedObjects(labels,mask.numpy()) + " " + self.instance_prompt[index % self.num_instance_images]
        prompt = PROMPTS_START[np.random.randint(0,2)] + prompt + ", high resolution, highly detailed"

        if (self.mask_blur) and (np.random.randint(1,10)>=4):
            ker = np.random.randint(1,100)*2-1
            q = np.random.randint(1,40)
            mask_blur = torch.Tensor(cv2.GaussianBlur(mask.numpy().astype(float), (ker,ker), q)).to(torch.float32)
            example["mask"] = mask_blur
        else:
            example["mask"] = mask  
            example["masked_image"] = masked_image 

        if (self.prop_empty_prompt>0) and (random.random()<self.prop_empty_prompt):
            example["prompt"] = ""
        else:
            example["prompt"] = prompt
        example["instance_prompt_ids"] = self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            class_image = self.image_transforms_resize_and_crop(class_image)
            example["class_images"] = self.image_transforms(class_image)
            example["class_PIL_images"] = class_image
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example
