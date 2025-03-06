import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, default_collate
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR
from torchvision.transforms import v2
from torchvision import datasets as dset
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights
from diffusers import AutoPipelineForText2Image
from datasets import load_from_disk
from medmnist import DermaMNIST, BloodMNIST
import os
import numpy as np
import shutil
from PIL import Image
import pandas as pd
import logging
from datetime import datetime
import random
import time
import copy
import json
import gc
import psutil
import scipy
import traceback
import csv
import sys
import pickle



############################################################################################################
# Seed and Logger Setup: 
############################################################################################################

seed = 420

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Optimize the performance of convolution operations in PyTorch by leveraging the best available algorithms for the given hardware and input sizes
torch.backends.cudnn.benchmark = True

# Create the logs directory if it doesn't exist
log_dir = '../storage/all_pipelines_logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Generate a filename based on the current time
base_time = datetime.now().strftime('%Y%m%d_%H%M')
current_time = base_time
log_filename = os.path.join(log_dir, f'{current_time}.log')

# If the log file already exists, append a number to current_time to avoid overwriting it
i = 1
while os.path.exists(log_filename):
    current_time = f'{base_time}_{i}'
    log_filename = os.path.join(log_dir, f'{current_time}.log')
    i += 1

# Set up logging
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w') # filemode='w' to overwrite the log file every time the pipeline is run
logger = logging.getLogger()



############################################################################################################
# Image Generation: 
############################################################################################################

def generate_random_image(image_size):
    return torch.rand(3, *image_size) * 2 - 1  # Random image with values in the range [-1, 1], as expected by the OFA model


def load_sd_model(pretrained_sd, finetuned_sd_path, finetuned_sd_weights, device, noise_sigma):
    logger.info("Loading the Stable Diffusion model...")
    pipeline = AutoPipelineForText2Image.from_pretrained(pretrained_sd, torch_dtype=torch.float16, use_safetensors=True, safety_checker = None, requires_safety_checker = False).to(device)
    
    if (finetuned_sd_path is not None) and (finetuned_sd_weights is not None):
        pipeline.load_lora_weights(finetuned_sd_path, weight_name=finetuned_sd_weights)
        logger.info("Finetuned SD loaded.\n")
    else:
        logger.info("NOT-finetuned SD loaded.\n")

    pipeline.scheduler.init_noise_sigma = noise_sigma
    if noise_sigma != 1:
        logger.info(f"Initial noise sigma (std) is set to {noise_sigma}, instead of default 1.\n")

    return pipeline


def generate_single_image_with_stable_diffusion(pipeline, text_prompt, sd_output_resolution, num_inference_steps, guidance_scale):
    width, height = sd_output_resolution
    image = pipeline(text_prompt, width=width, height=height, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

def generate_batch_images_with_stable_diffusion(pipeline, text_prompts, sd_output_resolution, num_inference_steps, guidance_scale):
    width, height = sd_output_resolution
    image = pipeline(text_prompts, width=width, height=height, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
    return image



############################################################################################################
# Dataset Pre-Processing:
############################################################################################################

def get_transform(dataset_name, image_size, huge_augment, horizontal_flip, random_crop, random_erasing, train=True):

    if dataset_name == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    elif dataset_name == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif dataset_name == 'oxfordpets':
        mean = [0.4717, 0.4499, 0.3837]
        std = [0.2726, 0.2634, 0.2794]
    elif dataset_name == 'stanfordcars':
        mean = [0.4708, 0.4602, 0.4550]
        std = [0.2892, 0.2882, 0.2968]
    elif dataset_name == 'food101':
        mean = [0.5450, 0.4435, 0.3436]
        std = [0.2695, 0.2719, 0.2766]
    elif dataset_name == 'tinyimagenet':
        mean = [0.4805, 0.4483, 0.3978]
        std = [0.2177, 0.2138, 0.2136]
    elif dataset_name == 'dermamnist': 
        mean = [0.7632, 0.5381, 0.5615]
        std = [0.0872, 0.1204, 0.1360]
    elif dataset_name == 'bloodmnist':
        mean = [0.7961, 0.6596, 0.6964]
        std = [0.2139, 0.2464, 0.0903]
    elif dataset_name == 'stl10':
        mean = [0.4467, 0.4398, 0.4066]
        std = [0.2185, 0.2159, 0.2183]
    elif dataset_name == 'imagenette':
        mean = [0.4625, 0.4580, 0.4295]
        std = [0.2351, 0.2287, 0.2372]
    elif dataset_name == 'caltech101':
        mean = [0.5418, 0.5209, 0.4857]
        std = [0.2389, 0.2378, 0.2376]
    elif dataset_name == 'imagewoof': 
        mean = [0.4861, 0.4560, 0.3938]
        std = [0.2207, 0.2145, 0.2166]
    else:
        raise TypeError(f"Unknown dataset: {dataset_name}.")

    transformations = [v2.ToTensor(), v2.Resize(image_size, interpolation=Image.BICUBIC, antialias=True)]

    if train:
        if horizontal_flip:
            transformations.append(v2.RandomHorizontalFlip())

        if huge_augment == 'trivial_augment':
            transformations.append(v2.TrivialAugmentWide())
        elif huge_augment == 'auto_augment':
            transformations.append(v2.AutoAugment())
        elif huge_augment == 'rand_augment':
            transformations.append(v2.RandAugment())
        elif huge_augment == 'aug_mix':
            transformations.append(v2.AugMix())

        if random_crop:
            padding_fraction = 0.10 
            new_padding = int(image_size[0] * padding_fraction)
            transformations.append(v2.RandomCrop(image_size[0], padding=new_padding))

    transformations.append(v2.Normalize(mean=mean, std=std))

    if train and random_erasing:
        transformations.append(v2.RandomErasing(value='random'))

    transform = v2.Compose(transformations)

    return transform



############################################################################################################
# Synthetic Training Dataset Creation (by generation or retrieval from disk):
############################################################################################################

class SyntheticTrainDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform, num_synthetic_samples, num_total_synthetic_datasets_used, dataset_replacements_already_done, mode):
        # Save here only the parameters used by __len__ and __getitem__ methods (because they only have self as argument); no need to save here all the parameters
        self.images_dir = images_dir
        self.transform = transform
        self.num_synthetic_samples = num_synthetic_samples
        self.dataset_replacements_already_done = dataset_replacements_already_done
        self.mode = mode
        self.metadata = []

        # Sanity check
        if mode not in ["generation", "sampling"]:
            raise ValueError(f"Invalid mode: {mode}. Supported modes are 'generation' and 'sampling'.")

        # Initialize or load self.metadata
        if mode == "sampling":
            if dataset_replacements_already_done == 0:
                # Load all samples from CSV
                all_samples = []
                with open(labels_csv, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        filename = row[0]
                        hard_label = int(row[1])
                        all_samples.append((filename, hard_label))
                # Check if there are enough samples for all resamplings (each time the subset of samples is different)
                total_samples_needed = num_synthetic_samples * (num_total_synthetic_datasets_used)
                if len(all_samples) < total_samples_needed:
                    raise ValueError(f"Total samples needed for all resamplings is {total_samples_needed}, "
                                     f"but the number of samples in the {images_dir} dataset is only {len(all_samples)}.")
                # Randomly select the samples needed for this and future resamplings
                self.metadata = random.sample(all_samples, total_samples_needed)
                # Garbage collection
                del all_samples
                torch.cuda.empty_cache()
                gc.collect()
                # Save the selected samples to a temporary file
                os.makedirs("temp_pipeline_files", exist_ok=True)
                with open(os.path.join("temp_pipeline_files", f"{current_time}_temp_samples_selected.pkl"), "wb") as f:
                    pickle.dump(self.metadata, f)
            else:
                # Load samples from the saved temporary file
                with open(os.path.join("temp_pipeline_files", f"{current_time}_temp_samples_selected.pkl"), "rb") as f:
                    self.metadata = pickle.load(f)

        elif mode == "generation":
            with open(labels_csv, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    filename = row[0]
                    hard_label = int(row[1])
                    self.metadata.append((filename, hard_label))

    def __len__(self):
        return self.num_synthetic_samples

    def __getitem__(self, idx):
        if self.mode == "generation":
            filename, hard_label = self.metadata[idx]
        elif self.mode == "sampling":
            filename, hard_label = self.metadata[idx + self.num_synthetic_samples * self.dataset_replacements_already_done]
        label = torch.tensor(hard_label)

        image_path = os.path.join(self.images_dir, filename)
        image = Image.open(image_path).convert("RGB")

        transformed_image = self.transform(image)
        del image

        return transformed_image, label


def get_synthetic_train_dataset(
    num_synthetic_samples,
    batch_gen_images,
    sd_output_resolution,
    classifier_input_resolution,
    num_inference_steps,
    guidance_scale,
    num_classes,
    sd_model,
    pretrained_sd,
    finetuned_sd_path,
    finetuned_sd_weights,
    huge_augment,
    horizontal_flip,
    random_crop,
    random_erasing,
    dataset_name,
    prompt_type,
    prompts_file,
    already_saved_synthetic_dataset_dir,
    num_total_synthetic_datasets_used,
    dataset_replacements_already_done,
):
    # In this branch we generate the synthetic dataset with Stable Diffusion
    if already_saved_synthetic_dataset_dir is None:
        mode = "generation"

        # Generate prompts and labels
        prompts, hard_labels = get_synthetic_labels_and_prompts(
            prompts_file=prompts_file,
            prompt_type=prompt_type,
            num_synthetic_samples=num_synthetic_samples
        )

        # Directory to save dataset
        synthetic_dataset_dir = os.path.join("../storage/synthetic_datasets_not_preprocessed", current_time)

        # Delete previous dataset if it exists (it happens if synthetic_dataset_epochs_replacement is set to a value smaller than num_epochs)
        if os.path.exists(synthetic_dataset_dir):
            shutil.rmtree(synthetic_dataset_dir)
            logger.info(f"Deleted the previous synthetic dataset at {synthetic_dataset_dir}: a new one will now be generated.\n")
        os.makedirs(synthetic_dataset_dir)

        # Some setup
        metadata = []
        num_digits = len(str(len(prompts))) # to save the images filename with the right lenght
        garbage_collection_interval = 5000

        logger.info("Generating the synthetic training images...\n")

        for i in range(0, len(prompts), batch_gen_images):
            # Generate a batch of images
            batch_prompts = prompts[i : i + batch_gen_images]
            generated_images = generate_batch_images_with_stable_diffusion(
                pipeline=sd_model,
                text_prompts=batch_prompts,
                sd_output_resolution=sd_output_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )

            # Save the images to disk
            for j, image in enumerate(generated_images):
                idx = i + j
                hard_label = hard_labels[idx]
                filename = f"image_{idx+1:0{num_digits}d}.png"
                filepath = os.path.join(synthetic_dataset_dir, filename)
                image.save(filepath)

                metadata.append((filename, hard_label))

                del image, filename, hard_label

                if (idx+1) % garbage_collection_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"Generated and saved {idx+1} synthetic images.")
                    logger.info(f"RAM usage: {get_memory_usage():.2f} GB \n")

            del generated_images, batch_prompts

        logger.info("All generated images have been saved on disk.\n")

        # Garbage collection
        del hard_labels, prompts
        torch.cuda.empty_cache()
        gc.collect()

        # Save filename and labels to a CSV file
        logger.info("Saving filenames and hard labels on a CSV file on disk...")
        labels_csv_path = os.path.join(synthetic_dataset_dir, "hard_labels.csv")
        with open(labels_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for filename, hard_label in metadata:
                writer.writerow([filename, hard_label])
        logger.info("CSV file saved on disk.\n")


        # Write in a text file the parameters used to generate the synthetic dataset
        with open(os.path.join(synthetic_dataset_dir, "dataset_generation_parameters.txt"), "w") as file:
            file.write(f"dataset_name: {dataset_name}\n")
            file.write(f"num_classes: {num_classes}\n")
            file.write(f"num_synthetic_samples: {num_synthetic_samples}\n")
            file.write(f"batch_gen_images: {batch_gen_images}\n")
            file.write(f"pretrained_sd: {pretrained_sd}\n")
            file.write(f"finetuned_sd_path: {finetuned_sd_path}\n")
            file.write(f"finetuned_sd_weights: {finetuned_sd_weights}\n")
            file.write(f"sd_output_resolution: {sd_output_resolution}\n")
            file.write(f"num_inference_steps: {num_inference_steps}\n")
            file.write(f"guidance_scale: {guidance_scale}\n")
            file.write(f"prompt_type: {prompt_type}\n")
            file.write(f"prompts_file: {prompts_file}\n")


    # If this other branch, we retrieve the synthetic dataset from a dataset already saved on disk
    else:
        mode = "sampling"

        logger.info("We will sample the synthetic dataset from a dataset already saved on disk, instead of generating it.\n")

        synthetic_dataset_dir = already_saved_synthetic_dataset_dir
        labels_csv_path = os.path.join(synthetic_dataset_dir, "hard_labels.csv")


    # Retrieve the transformation to preprocess the synthetic images for the classifier
    transform = get_transform(dataset_name, classifier_input_resolution, huge_augment, horizontal_flip, random_crop, random_erasing, train=True)

    logger.info("Instantiating the Synthetic Image Dataset class...")

    # Create the synthetic dataset for training the classifier
    synthetic_dataset = SyntheticTrainDataset(
        images_dir=synthetic_dataset_dir,
        labels_csv=labels_csv_path,
        transform=transform,
        num_synthetic_samples=num_synthetic_samples,
        num_total_synthetic_datasets_used=num_total_synthetic_datasets_used,
        dataset_replacements_already_done=dataset_replacements_already_done,
        mode=mode,
    )

    logger.info("Synthetic dataset instantiated.\n")

    torch.cuda.empty_cache()
    gc.collect()

    # RAM usage after generating the synthetic dataset
    logger.info(f"RAM usage after getting the synthetic dataset: {get_memory_usage():.2f} GB\n")

    return synthetic_dataset_dir, synthetic_dataset



############################################################################################################
# Real Training/Testing Datasets Retrieval:
############################################################################################################

class OxfordPetsDataset(Dataset):
    def __init__(self, root='../datasets/oxfordpets', split='train', transform=None):
        self.split = split
        self.transform = transform
        self.data_dir = root+'/train85.pth' if split == 'train' else root+'/test15.pth'
        self.data = torch.load(self.data_dir)
        self.classes = sorted(set(label.item() for _, label in self.data)) # -> 0, 1, 2, ..., 36

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]

        if self.transform:
            img = self.transform(img)

        return img, label


class TinyImageNetDataset(Dataset):
    def __init__(self, root='../datasets/tinyimagenet', split='train', transform=None):
        self.split = split
        self.transform = transform
        self.data_dir = root+'/train' if split == 'train' else root+'/valid'
        self.data = load_from_disk(self.data_dir)
        self.classes = self.data.features['label'].names
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        img = example['image']
        label = example['label']

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


class Caltech101Dataset(Dataset):
    def __init__(self, root='../datasets/caltech101', split='train', transform=None):
        self.split = split
        self.transform = transform
        self.data_dir = root+'/train' if split == 'train' else root+'/test'
        self.data = load_from_disk(self.data_dir)
        self.classes = self.data.features['label'].names
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        img = example['image']
        label = example['label']

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


class ImagewoofDataset(Dataset):
    def __init__(self, root='../datasets/imagewoof', split='train', transform=None):
        self.split = split
        self.transform = transform
        self.data_dir = root+'/train' if split == 'train' else root+'/validation'
        self.data = load_from_disk(self.data_dir)
        self.classes = self.data.features['label'].names
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        img = example['image']
        label = example['label']

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label



def get_real_dataset(dataset_name, image_size, huge_augment, horizontal_flip, random_crop, random_erasing):
    
    logger.info(f'Dataset {dataset_name} loading and processing...')

    training_transformations = get_transform(dataset_name, image_size, huge_augment, horizontal_flip, random_crop, random_erasing, train=True)
    test_transformations = get_transform(dataset_name, image_size, huge_augment, horizontal_flip, random_crop, random_erasing, train=False)

    if dataset_name == 'cifar10':
        train_data = dset.CIFAR10(root='../datasets/cifar10', train=True, transform=training_transformations, download=True)
        test_data = dset.CIFAR10(root='../datasets/cifar10', train=False, transform=test_transformations, download=True)
    elif dataset_name == 'cifar100':
        train_data = dset.CIFAR100(root='../datasets/cifar100', train=True, transform=training_transformations, download=True)
        test_data = dset.CIFAR100(root='../datasets/cifar100', train=False, transform=test_transformations, download=True)
    elif dataset_name == 'oxfordpets':
        train_data = OxfordPetsDataset(root='../datasets/oxfordpets', split='train', transform=training_transformations)
        test_data = OxfordPetsDataset(root='../datasets/oxfordpets', split='test', transform=test_transformations)
    elif dataset_name == 'stanfordcars':
        train_data = dset.StanfordCars(root='../datasets/stanfordcars', split='train', transform=training_transformations, download=False) # download does not work for this dataset, it's only for backward compatibility
        test_data = dset.StanfordCars(root='../datasets/stanfordcars', split='test', transform=test_transformations, download=False) # download does not work for this dataset, it's only for backward compatibility
    elif dataset_name == 'food101':
        train_data = dset.Food101(root='../datasets/food101', split='train', transform=training_transformations, download=True)
        test_data = dset.Food101(root='../datasets/food101', split='test', transform=test_transformations, download=True)
    elif dataset_name == 'tinyimagenet':
        train_data = TinyImageNetDataset(root='../datasets/tinyimagenet', split='train', transform=training_transformations)
        test_data = TinyImageNetDataset(root='../datasets/tinyimagenet', split='valid', transform=test_transformations) # valid is the test set for Tiny ImageNet huggingface dataset
    elif dataset_name == 'dermamnist':
        train_data = DermaMNIST(root='../datasets/dermamnist', split='train', size=224, as_rgb=True, transform=training_transformations, download=True)
        test_data = DermaMNIST(root='../datasets/dermamnist', split='test', size=224, as_rgb=True, transform=test_transformations, download=True)
    elif dataset_name == 'bloodmnist':
        train_data = BloodMNIST(root='../datasets/bloodmnist', split='train', size=224, as_rgb=True, transform=training_transformations, download=True)
        test_data = BloodMNIST(root='../datasets/bloodmnist', split='test', size=224, as_rgb=True, transform=test_transformations, download=True)
    elif dataset_name == 'stl10':
        train_data = dset.STL10(root='../datasets/stl10', split='train', transform=training_transformations, download=True)
        test_data = dset.STL10(root='../datasets/stl10', split='test', transform=test_transformations, download=True)
    elif dataset_name == 'imagenette':
        train_data = dset.Imagenette(root='../datasets/imagenette', split='train', transform=training_transformations, download=False) # for this dataset, download=True returns an error if the dataset is already downloaded
        test_data = dset.Imagenette(root='../datasets/imagenette', split='val', transform=test_transformations, download=False) # for this dataset, download=True returns an error if the dataset is already downloaded
    elif dataset_name == 'caltech101':
        train_data = Caltech101Dataset(root='../datasets/caltech101', split='train', transform=training_transformations)
        test_data = Caltech101Dataset(root='../datasets/caltech101', split='test', transform=test_transformations)
    elif dataset_name == 'imagewoof':
        train_data = ImagewoofDataset(root='../datasets/imagewoof', split='train', transform=training_transformations)
        test_data = ImagewoofDataset(root='../datasets/imagewoof', split='validation', transform=test_transformations)
    else:
        raise TypeError(f"Unknown dataset: {dataset_name}.")


    logger.info(f'Dataset {dataset_name} loaded and processed!\n')
        
    return train_data, test_data



############################################################################################################
# Hard Labels and Prompts for the (potential) synthetic dataset generation:
############################################################################################################

def get_synthetic_labels_and_prompts(prompts_file, prompt_type, num_synthetic_samples):
    if prompts_file.endswith('.txt'): # for Claude
        return retrieve_from_claude_txt_file(prompts_file, prompt_type, num_synthetic_samples)
    elif prompts_file.endswith('.json'): # for Blip-2
        return retrieve_from_blip2_json_file(prompts_file, prompt_type, num_synthetic_samples)
    else:
        raise ValueError(f"Unsupported prompts file type: {prompts_file}. Supported types are .txt (for Claude) and .json (for Blip-2)")


def parse_claude_txt_file(file_path):
    class_names = []
    class_descriptions = []
    with open(file_path, 'r') as file: # File format: "class_name: class_description"
        for line in file:
            class_name, class_description = line.strip().split(': ', 1)
            class_names.append(class_name.strip()) # strip() to remove leading/trailing whitespaces
            class_descriptions.append(class_description.strip())
    # Ensure class_name are in alphabetical order (assuming that in the corresponding real dataset the classes are numbered alphabetically), 
    # otherwise the numeric labels will be assigned in a different order!
    class_names, class_descriptions = zip(*sorted(zip(class_names, class_descriptions)))
    return class_names, class_descriptions


def retrieve_from_claude_txt_file(txt_file_path, prompt_type, num_synthetic_samples):
    class_names, class_descriptions = parse_claude_txt_file(txt_file_path) # class_names are sorted alphabetically

    num_classes = len(class_names)
    samples_per_class = num_synthetic_samples // num_classes
    remainder = num_synthetic_samples % num_classes

    prompts = []
    labels = []

    # Generate samples for each class
    for class_idx in range(num_classes):
        for _ in range(samples_per_class):
            if prompt_type == "only_className":
                prompt = class_names[class_idx]
            elif prompt_type == "only_text":
                prompt = class_descriptions[class_idx]
            elif prompt_type == "className_and_text":
                prompt = f"{class_names[class_idx]}: {class_descriptions[class_idx]}"
            else:
                raise ValueError(f"Not implemented prompt_type: {prompt_type}")
            prompts.append((prompt, class_idx)) # double parenthesis to append a tuple to the list instead of two separate elements

    # Distribute the remainder samples
    for i in range(remainder):
        class_idx = i % num_classes
        if prompt_type == "only_className":
            prompt = class_names[class_idx]
        elif prompt_type == "only_text":
            prompt = class_descriptions[class_idx]
        elif prompt_type == "className_and_text":
            prompt = f"{class_names[class_idx]}: {class_descriptions[class_idx]}"
        else:
            raise ValueError(f"Not implemented prompt_type: {prompt_type}")
        prompts.append((prompt, class_idx))

    # Shuffle the combined list to randomize the order
    random.shuffle(prompts)

    # Separate the prompts and labels
    prompts, labels = zip(*prompts)
    prompts = list(prompts)
    labels = list(labels)         

    return prompts, labels



def retrieve_from_blip2_json_file(json_file_path, prompt_type, num_synthetic_samples):
    with open(json_file_path, 'r') as f:
        captions_data = json.load(f)

    # In the Hugging Face "imagenette" and "imagewoof" datasets, classes are not indexed alphabetically, so we list them in the correct order
    if "imagenette" in json_file_path:
        class_names = ["tench", "English springer", "cassette player", "chain saw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
    elif "imagewoof" in json_file_path:
        class_names = ["Australian terrier", "Border terrier", "Samoyed", "Beagle", "Shih-Tzu", "English foxhound", "Rhodesian ridgeback", "Dingo", "Golden retriever", "Old English sheepdog"]
    else:
        class_names = sorted(captions_data.keys()) # class_names are sorted alphabetically
    num_classes = len(class_names)
    samples_per_class = num_synthetic_samples // num_classes
    remainder = num_synthetic_samples % num_classes

    prompts = []
    labels = []

    # Generate samples for each class
    for class_idx, class_name in enumerate(class_names):
        captions = captions_data[class_name]
        for i in range(samples_per_class):
            if prompt_type == "only_className":
                prompt = class_name
            elif prompt_type == "only_text":
                prompt = captions[i % len(captions)]
            elif prompt_type == "className_and_text":
                prompt = f"{class_name}: {captions[i % len(captions)]}"
            else:
                raise ValueError(f"Not implemented prompt_type: {prompt_type}")
            prompts.append((prompt, class_idx))

    # Distribute the remainder samples
    for i in range(remainder):
        class_idx = i % num_classes
        class_name = class_names[class_idx]
        captions = captions_data[class_name]
        offset = samples_per_class % len(captions)
        if prompt_type == "only_className":
            prompt = class_name
        elif prompt_type == "only_text":
            prompt = captions[(i + offset) % len(captions)]
        elif prompt_type == "className_and_text":
            prompt = f"{class_name}: {captions[(i + offset) % len(captions)]}"
        else:
            raise ValueError(f"Not implemented prompt_type: {prompt_type}")
        prompts.append((prompt, class_idx))

    # Shuffle the combined list to randomize the order
    random.shuffle(prompts)

    # Separate the prompts and labels
    prompts, labels = zip(*prompts)
    prompts = list(prompts)
    labels = list(labels)

    return prompts, labels



############################################################################################################
# Teacher/Student Classifiers:
############################################################################################################

def extract_mbnv3_ofa(config, num_classes):
    
    logger.info('MobilNetV3 Classifier loading...')
    
    split_pieces = config.split("_")
    first_el = split_pieces.pop(0)

    d = [int(piece.split("-")[0]) for i, piece in enumerate(split_pieces) if i % 4 == 0]
    k = [int(piece.split("-")[1]) if piece.split("-")[1] != '0' else 3 for piece in split_pieces]
    e = [int(piece.split("-")[2]) if piece.split("-")[2] != '0' else 3 for piece in split_pieces]

    super_net_name = 'ofa_supernet_mbv3_w10' if first_el == '10' else 'ofa_supernet_mbv3_w12'
    super_net = torch.hub.load('mit-han-lab/once-for-all', super_net_name, pretrained=True, verbose=False).eval() 
    super_net.set_active_subnet(d=d, e=e, ks=k)
    model = super_net.get_active_subnet(preserve_weight=True)
    
    in_features = model.classifier.linear.in_features
    model.classifier = nn.Sequential(nn.Linear(in_features, num_classes))
    
    logger.info('MobilNetV3 Classifier loaded!\n')
    
    return model


class CustomEfficientNetV2(nn.Module):
    def __init__(self, size, num_classes, pretrained=True):
        assert size in ['s', 'm', 'l'] # safe check
        super(CustomEfficientNetV2, self).__init__()
        
        if pretrained:
            if size == 's':
                weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
                self.model = efficientnet_v2_s(weights=weights)
            elif size == 'm':
                weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
                self.model = efficientnet_v2_m(weights=weights)
            elif size == 'l':
                weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
                self.model = efficientnet_v2_l(weights=weights)
        else:
            if size == 's':
                self.model = efficientnet_v2_s(weights=None)
            elif size == 'm':
                self.model = efficientnet_v2_m(weights=None)
            elif size == 'l':
                self.model = efficientnet_v2_l(weights=None)
        
        num_features = self.model.classifier[1].in_features
        
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)



############################################################################################################
# Auxiliary Functions for Training and Testing:
############################################################################################################

def learning_rate_scheduling(optimizer, scheduler_name=None, warmup=False, warmup_epochs=0, epochs=100):
    if scheduler_name is not None:
        scheduler_list = []

        if warmup:
            lr_lambda = lambda epoch: (epoch / warmup_epochs) + 1e-5
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            scheduler_list.append(warmup_scheduler)
            milestones = [warmup_epochs]
        else:
            milestones = []

        if scheduler_name == 'CosineAnnealingLR':
            scheduler_lr = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=0, last_epoch=-1, verbose=False)
            scheduler_list.append(scheduler_lr)
            
        if scheduler_list:
            scheduler = SequentialLR(optimizer, schedulers=scheduler_list, milestones=milestones)
        else:
            scheduler = None
    else:
        scheduler = None

    return scheduler


def train_epoch(
    model, 
    train_dataloader,
    dataset_name, 
    criterion, 
    optimizer, 
    scaler, 
    mixed_precision, 
    scheduler,
    device,
    cutmix_or_mixup,
    use_soft_labels,
    teacher_classifier
):
    model.train()
    total_loss, correct_predictions, total_samples, nan_encountered = 0.0, 0, 0, False
    
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.float().to(device), labels.to(device)
        optimizer.zero_grad()

        # Squeeze (if not already done in collate_fn) for dermamnist or bloodmnist: from shape [batch_size, 1] to shape [batch_size]
        if not cutmix_or_mixup and dataset_name in ['dermamnist', 'bloodmnist'] and labels.dim() == 2:
            labels = labels.squeeze(1)

        # Produce the soft-label (if requested) for the synthetic train dataset
        if use_soft_labels and teacher_classifier is not None:
            with torch.no_grad():
                logits = teacher_classifier(inputs)
                labels = nn.Softmax(dim=1)(logits)
        
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss).any():
                logger.info("==> Encountered NaN value in loss.")
                return 0, 0, True
            
            scaler.scale(loss).backward() if mixed_precision else loss.backward()
            scaler.step(optimizer) if mixed_precision else optimizer.step()
            scaler.update() if mixed_precision else None
        
        total_loss += loss.item()
        _, int_predictions = torch.max(outputs, 1) # find the class with the highest probability and save its index (the class integer label)
        if cutmix_or_mixup or use_soft_labels: 
            _, labels = torch.max(labels, 1) # find the class with the highest probability and save its index (the class integer label)
        correct_predictions += (int_predictions == labels).sum().item()
        total_samples += labels.size(0)
    
    if scheduler:
        scheduler.step()
    
    average_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy, nan_encountered


def test_epoch(
    model,
    test_dataloader,
    dataset_name,
    criterion,
    device
):
    model.eval()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.float().to(device), labels.to(device)

            # Squeeze only for dermamnist or bloodmnist: from shape [batch_size, 1] to shape [batch_size]
            if dataset_name in ['dermamnist', 'bloodmnist'] and labels.dim() == 2:
                labels = labels.squeeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, int_predictions = torch.max(outputs, 1)
            correct_predictions += (int_predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    average_loss = total_loss / len(test_dataloader)
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy
    

def get_memory_usage():
    process = psutil.Process()
    ram_gb = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert bytes to GB
    return ram_gb



############################################################################################################
# Main Function, for classifiers evaluation, combining all the above functions:
############################################################################################################

def run_script(
    dataset_name,
    finetuned_sd_path,
    prompts_file,
    synthetic_to_real_train_dataset_ratio,
    save_synthetic_dataset,
    already_saved_synthetic_dataset_dir,
    use_soft_labels,
    efficientnet_teacher_path,
    ofa_config,
    use_also_real_classifier,
    trained_real_classifier_weights,
    epochs,
    synthetic_dataset_epochs_replacement,
    pretrained_sd="stabilityai/stable-diffusion-2",
    num_inference_steps=20,
    guidance_scale=2,
    noise_sigma=1,
    sd_output_resolution=(224, 224),
    finetuned_sd_weights="pytorch_lora_weights.safetensors",
    prompt_type="className_and_text",
    batch_gen_images=32,
    classifier_input_resolution=(224, 224),
    trained_synthetic_classifier_weights=None,
    horizontal_flip=True,
    huge_augment='aug_mix',
    random_crop=True,
    random_erasing=False,
    cutmix=False,
    mixup=True,
    optimizer_name='AdamW',
    label_smoothing=0.1,
    learning_rate=0.001,
    batch_size=96,
    weight_decay=0.00005,
    scheduler_name='CosineAnnealingLR',
    warmup=False,
    warmup_epochs=0,
    patience=30,
    mixed_precision=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):

    # Try-except-finally block to catch any potential error and log it
    try:

        # It makes no sense to generate and save a synthetic dataset if we are sampling it from a dataset already saved on disk
        if save_synthetic_dataset and already_saved_synthetic_dataset_dir:
            raise ValueError("You forgot to set to False/None one of the two parameters 'save_synthetic_dataset' and 'already_saved_synthetic_dataset_dir': it makes no sense to have both of them at the same time.")

        # Set epochs=0 only if you want to generate and save a synthetic dataset without training the classifiers
        if epochs == 0 and save_synthetic_dataset == False:
            raise ValueError("You set 'epochs' to 0 but you didn't set 'save_synthetic_dataset' to True.")

        # Check that a pretrained real classifier is given if soft labels were requested
        if use_soft_labels and (use_also_real_classifier == False or trained_real_classifier_weights==None) and efficientnet_teacher_path==None:
            raise ValueError("Soft labels can be generated only if a trained real classifier is provided.")

        # Write on the log file the parameters of this main function (referred to as "pipeline")
        logger.info("Pipeline Parameters:\n")
        logger.info(f"seed: {seed}")
        for param_name, param_value in locals().items(): # locals() returns a dictionary with the current local variables
            logger.info(f"{param_name}: {param_value}")
        logger.info("---------------------------------------------------------\n")

        # Initialize these variables to avoid errors later
        model_real = None
        synthetic_dataset_dir = None

        # Record the start time of the pipeline
        start_time_pipeline = time.time()

        # RAM usage at the beginning
        logger.info(f"RAM usage at the beginning: {get_memory_usage():.2f} GB\n")

        # Load the real dataset
        train_dataset_real, test_dataset_real = get_real_dataset(
            dataset_name=dataset_name,
            image_size=classifier_input_resolution, 
            horizontal_flip = horizontal_flip, 
            huge_augment = huge_augment, 
            random_crop = random_crop, 
            random_erasing = random_erasing
        )

        # RAM usage after retrieving the real dataset
        logger.info(f"RAM usage after retrieving the real dataset from disk: {get_memory_usage():.2f} GB\n")

        # Get the number of classes and the number of synthetic samples we want to have
        if dataset_name == 'dermamnist': # since the medmnist datasets don't have the "classes" attribute
            num_classes = 7
        elif dataset_name == 'bloodmnist': # since the medmnist datasets don't have the "classes" attribute
            num_classes = 8
        else:
            num_classes = len(train_dataset_real.classes)
        num_synthetic_samples = int(len(train_dataset_real) * synthetic_to_real_train_dataset_ratio)
        logger.info(f"Lenght of the synthetic dataset: {num_synthetic_samples}\n")

        # Compute how many times we will change the synthetic dataset
        num_total_synthetic_datasets_used = 1 + epochs // synthetic_dataset_epochs_replacement
        logger.info(f"A total of {num_total_synthetic_datasets_used} synthetic datasets will be generated/sampled.\n")

        # Save the numbers of synthetic dataset replacements already done
        dataset_replacements_already_done = 0

        # Some data augmentation parameters for the training synthetic (and real) datasets
        if cutmix or mixup:
            if cutmix and mixup:
                advanced_transform = v2.RandomChoice([v2.CutMix(num_classes=num_classes), v2.MixUp(num_classes=num_classes)])
            elif cutmix:
                advanced_transform = v2.CutMix(num_classes=num_classes)
            else:
                advanced_transform = v2.MixUp(num_classes=num_classes)

            def collate_fn(batch):
                images, labels = default_collate(batch)
                if dataset_name in ['dermamnist', 'bloodmnist'] and labels.dim() == 2:
                    labels = labels.squeeze(1)
                return advanced_transform(images, labels)

            cutmix_or_mixup = True
        else:
            cutmix_or_mixup = False

        
        # Real training dataset loading
        if use_also_real_classifier:
            logger.info("Creating the DataLoader of the real train dataset...")
            if cutmix_or_mixup:
                train_dataloader_real = DataLoader(train_dataset_real, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, collate_fn=collate_fn)
            else:
                train_dataloader_real = DataLoader(train_dataset_real, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
            logger.info("Real Train DataLoader created.\n")

        # Real testing dataset loading
        logger.info("Creating the DataLoader of the real test dataset...")
        test_dataloader = DataLoader(test_dataset_real, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
        logger.info("Real Test DataLoader created.\n")

        # RAM usage after loading the real datasets
        logger.info(f"RAM usage after loading the real datasets: {get_memory_usage():.2f} GB\n")

        # Classifier setup
        model_synthetic = extract_mbnv3_ofa(ofa_config, num_classes)
        model_synthetic = model_synthetic.to(device)
        if use_also_real_classifier:
            model_real = extract_mbnv3_ofa(ofa_config, num_classes)
            model_real = model_real.to(device)

        # RAM usage after loading the classifiers
        logger.info(f"RAM usage after loading the classifier(s): {get_memory_usage():.2f} GB\n")

        # Load the trained synthetic classifier's weights if specified
        if trained_synthetic_classifier_weights is not None:
            logger.info("Loading the trained synthetic classifier weights...")
            model_synthetic.load_state_dict(torch.load(trained_synthetic_classifier_weights))
            logger.info("Trained synthetic classifier weights loaded.\n")
            skip_training_synthetic = True
        else:
            skip_training_synthetic = False

        # Load the trained real classifier's weights if specified
        if use_also_real_classifier and trained_real_classifier_weights is not None:
            logger.info("Loading the trained real classifier weights...")
            model_real.load_state_dict(torch.load(trained_real_classifier_weights))
            logger.info("Trained real classifier weights loaded.\n")
            skip_training_real = True   
        else:
            skip_training_real = False

        # Load a teacher classifier (if specified) to generate soft labels
        if use_soft_labels and efficientnet_teacher_path is not None:
            logger.info("Loading the weights of the teacher classifier (different from OFA)...")
            teacher_classifier_size = efficientnet_teacher_path.split("/")[-1] # path is like "../storage/trained_efficientnet/cifar10/s"
            teacher_classifier = CustomEfficientNetV2(size=teacher_classifier_size, num_classes=num_classes)
            teacher_classifier = teacher_classifier.to(device)
            teacher_classifier.load_state_dict(torch.load(os.path.join(efficientnet_teacher_path, "best_model.pth")))
            teacher_classifier.eval() 
            logger.info("Teacher classifier weights loaded.\n")
        elif use_soft_labels and efficientnet_teacher_path is None:
            teacher_classifier = model_real
            teacher_classifier.eval()
            logger.info("The teacher classifier is a trained OFA classifier.\n")
        else:
            teacher_classifier = None

        # Some training setup
        scaler = torch.cuda.amp.GradScaler()
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if warmup:
            epochs += warmup_epochs

        # Synthetic classifier setup
        if optimizer_name == 'AdamW':
            optimizer_synthetic = optim.AdamW(model_synthetic.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer_synthetic = optim.SGD(model_synthetic.parameters(), lr=learning_rate, weight_decay=weight_decay, nesterov=True, momentum=0.9)
        scheduler_synthetic = learning_rate_scheduling(optimizer_synthetic, scheduler_name, warmup, warmup_epochs, epochs)
        best_accuracy_synthetic = 0.0
        best_model_synthetic = None
        no_improvement_epochs_synthetic = 0
        history_synthetic = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'learning_rate': [],
            'epoch_time': []
        }

        # Real classifier setup
        if use_also_real_classifier:
            if optimizer_name == 'AdamW':
                optimizer_real = optim.AdamW(model_real.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                optimizer_real = optim.SGD(model_real.parameters(), lr=learning_rate, weight_decay=weight_decay, nesterov=True, momentum=0.9)
            scheduler_real = learning_rate_scheduling(optimizer_real, scheduler_name, warmup, warmup_epochs, epochs)
            best_accuracy_real = 0.0
            best_model_real = None
            no_improvement_epochs_real = 0
            history_real = {
                'epoch': [],
                'train_loss': [],
                'train_accuracy': [],
                'test_loss': [],
                'test_accuracy': [],
                'learning_rate': [],
                'epoch_time': []
            }

        # Finetuned SD loading (mode: generation)
        if already_saved_synthetic_dataset_dir is None:
            sd_model = load_sd_model(pretrained_sd=pretrained_sd, finetuned_sd_path=finetuned_sd_path, finetuned_sd_weights=finetuned_sd_weights, device=device, noise_sigma=noise_sigma)
            logger.info(f"RAM usage after loading the SD model: {get_memory_usage():.2f} GB\n")
        else: # mode: sampling
            sd_model = None
            batch_gen_images = None


        # Training loop control variables
        stop_training_synthetic = False
        stop_training_real = False


        # If i have set epochs=0, i just generate and save the synthetic dataset and then stop without training the classifiers
        if epochs == 0:

            logger.info("The training/testing loop will not start because epochs=0.\n")
            skip_training_synthetic = True
            skip_training_real = True

            synthetic_dataset_dir, synthetic_dataset = get_synthetic_train_dataset(
                sd_output_resolution=sd_output_resolution,
                classifier_input_resolution=classifier_input_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_classes=num_classes,
                num_synthetic_samples=num_synthetic_samples,
                batch_gen_images=batch_gen_images,
                sd_model=sd_model,
                pretrained_sd=pretrained_sd,
                finetuned_sd_path=finetuned_sd_path,
                finetuned_sd_weights=finetuned_sd_weights,
                huge_augment=huge_augment,
                random_crop=random_crop,
                random_erasing=random_erasing,
                horizontal_flip=horizontal_flip,
                dataset_name=dataset_name,
                prompt_type=prompt_type,
                prompts_file=prompts_file,
                already_saved_synthetic_dataset_dir=already_saved_synthetic_dataset_dir,
                num_total_synthetic_datasets_used=num_total_synthetic_datasets_used,
                dataset_replacements_already_done=dataset_replacements_already_done
            )
   

        # Make the epochs loop start only if at least one classifier must be trained
        elif (not skip_training_synthetic) or (use_also_real_classifier and not skip_training_real):

            execute_only_one_epoch = True

            # Training and Testing loop
            for epoch in range(epochs): # from 0 to epochs-1

                # Train and test the synthetic classifier
                if not skip_training_synthetic and not stop_training_synthetic:

                    # Get a new synthetic dataset at epoch 0 and then every 'synthetic_dataset_epochs_replacement' epochs
                    if epoch % synthetic_dataset_epochs_replacement == 0:

                        if dataset_replacements_already_done > 0:
                            logger.info(f"Replacing the synthetic dataset for the {dataset_replacements_already_done} time:\n")

                        synthetic_dataset_dir, synthetic_dataset = get_synthetic_train_dataset( # Save also the synthetic dataset directory to potentially delete it later
                            sd_output_resolution=sd_output_resolution,
                            classifier_input_resolution=classifier_input_resolution,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            num_classes=num_classes,
                            num_synthetic_samples=num_synthetic_samples,
                            batch_gen_images=batch_gen_images,
                            sd_model=sd_model,
                            pretrained_sd=pretrained_sd, # just for logging purposes
                            finetuned_sd_path=finetuned_sd_path, # just for logging purposes
                            finetuned_sd_weights=finetuned_sd_weights, # just for logging purposes
                            huge_augment=huge_augment,
                            random_crop=random_crop,
                            random_erasing=random_erasing,
                            horizontal_flip=horizontal_flip,
                            dataset_name=dataset_name,
                            prompt_type=prompt_type,
                            prompts_file=prompts_file,
                            already_saved_synthetic_dataset_dir=already_saved_synthetic_dataset_dir,
                            num_total_synthetic_datasets_used=num_total_synthetic_datasets_used,
                            dataset_replacements_already_done=dataset_replacements_already_done
                        )

                        dataset_replacements_already_done += 1

                        logger.info("Creating the DataLoader of the synthetic train dataset...")
                        if cutmix_or_mixup and not use_soft_labels: # Disable cutmix/mixup if soft labels are used (documentation says that cutmix/mixup require integer labels)
                            train_dataloader_synthetic = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, collate_fn=collate_fn)
                        else:
                            train_dataloader_synthetic = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
                        logger.info("Synthetic DataLoader created.\n")

                    if epoch == 0:
                        logger.info("The epochs loop starts now!\n")
                        
                    start_time_synthetic = time.time()

                    train_loss_synthetic, train_accuracy_synthetic, nan_encountered_synthetic = train_epoch(
                        model=model_synthetic,
                        train_dataloader=train_dataloader_synthetic,
                        dataset_name = dataset_name,
                        criterion=criterion,
                        optimizer=optimizer_synthetic,
                        scaler=scaler,
                        mixed_precision=mixed_precision,
                        scheduler=scheduler_synthetic,
                        device=device,
                        cutmix_or_mixup=cutmix_or_mixup, 
                        use_soft_labels=use_soft_labels,
                        teacher_classifier=teacher_classifier
                    )
                    
                    if nan_encountered_synthetic:
                        train_loss_synthetic, train_accuracy_synthetic, nan_encountered_synthetic = train_epoch(
                            model=model_synthetic,
                            train_dataloader=train_dataloader_synthetic,
                            dataset_name = dataset_name,
                            criterion=criterion,
                            optimizer=optimizer_synthetic,
                            scaler=scaler,
                            mixed_precision=False,
                            scheduler=scheduler_synthetic,
                            device=device,
                            use_soft_labels=use_soft_labels,
                            teacher_classifier=teacher_classifier
                        )
                            
                    test_loss_synthetic, test_accuracy_synthetic = test_epoch(
                        model=model_synthetic,
                        test_dataloader=test_dataloader,
                        dataset_name = dataset_name,
                        criterion=criterion,
                        device=device
                    )

                    scheduler_last_lr_synthetic = scheduler_synthetic.get_last_lr()[0] if scheduler_synthetic is not None else learning_rate
                    epoch_time_synthetic = time.time() - start_time_synthetic

                    history_synthetic['epoch'].append(epoch + 1)
                    history_synthetic['train_loss'].append(train_loss_synthetic)
                    history_synthetic['train_accuracy'].append(train_accuracy_synthetic)
                    history_synthetic['test_loss'].append(test_loss_synthetic)
                    history_synthetic['test_accuracy'].append(test_accuracy_synthetic)
                    history_synthetic['learning_rate'].append(scheduler_last_lr_synthetic)
                    history_synthetic['epoch_time'].append(epoch_time_synthetic)

                    logger.info("--- Synthetic Classifier --- :")
                    logger.info(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time_synthetic:.2f}s - "
                    f"Train Loss: {train_loss_synthetic:.4f} - Train Acc: {train_accuracy_synthetic:.4f} - "
                    f"Test Loss: {test_loss_synthetic:.4f} - Test Acc: {test_accuracy_synthetic:.4f} - "
                    f"LR: {scheduler_last_lr_synthetic:.6f}")

                    if test_accuracy_synthetic > best_accuracy_synthetic:
                        best_accuracy_synthetic = test_accuracy_synthetic
                        best_model_synthetic = copy.deepcopy(model_synthetic)
                        no_improvement_epochs_synthetic = 0
                    else:
                        no_improvement_epochs_synthetic += 1
                        if no_improvement_epochs_synthetic >= patience:
                            logger.info(f"Early stopping at epoch {epoch+1} for Synthetic Classifier due to no improvement in test accuracy for {patience} consecutive epochs.\n")
                            stop_training_synthetic = True

                # Test the already trained synthetic classifier
                elif skip_training_synthetic and execute_only_one_epoch:
                    logger.info("\nSkipping the training loop for the synthetic classifier as it is already trained.\n")
                    start_time_synthetic = time.time()

                    test_loss_synthetic, test_accuracy_synthetic = test_epoch(
                        model=model_synthetic,
                        test_dataloader=test_dataloader,
                        dataset_name = dataset_name,
                        criterion=criterion,
                        device=device
                    )

                    epoch_time_synthetic = time.time() - start_time_synthetic

                    logger.info("--- Pre-Trained Synthetic Classifier Testing --- :")
                    logger.info(f"Epoch Time: {epoch_time_synthetic:.2f}s - "
                    f"Test Loss: {test_loss_synthetic:.4f} - Test Acc: {test_accuracy_synthetic:.4f} - \n")

                    # Stop the epochs loop just for this classifier since it is already trained, while the real classifier is not
                    execute_only_one_epoch = False
                


                # Train and test the real classifier
                if use_also_real_classifier and not skip_training_real and not stop_training_real:
                    start_time_real = time.time()

                    train_loss_real, train_accuracy_real, nan_encountered_real = train_epoch(
                        model=model_real,
                        train_dataloader=train_dataloader_real,
                        dataset_name = dataset_name,
                        criterion=criterion,
                        optimizer=optimizer_real,
                        scaler=scaler,
                        mixed_precision=mixed_precision,
                        scheduler=scheduler_real,
                        device=device,
                        cutmix_or_mixup=cutmix_or_mixup,
                        use_soft_labels=False,
                        teacher_classifier=None
                    )
                    
                    if nan_encountered_real:
                        train_loss_real, train_accuracy_real, nan_encountered_real = train_epoch(
                            model=model_real,
                            train_dataloader=train_dataloader_real,
                            dataset_name = dataset_name,
                            criterion=criterion,
                            optimizer=optimizer_real,
                            scaler=scaler,
                            mixed_precision=False,
                            scheduler=scheduler_real,
                            device=device,
                            use_soft_labels=False,
                            teacher_classifier=None
                        )
                        
                    test_loss_real, test_accuracy_real = test_epoch(
                        model=model_real,
                        test_dataloader=test_dataloader,
                        dataset_name = dataset_name,
                        criterion=criterion,
                        device=device
                    )

                    scheduler_last_lr_real = scheduler_real.get_last_lr()[0] if scheduler_real is not None else learning_rate
                    epoch_time_real = time.time() - start_time_real

                    history_real['epoch'].append(epoch + 1)
                    history_real['train_loss'].append(train_loss_real)
                    history_real['train_accuracy'].append(train_accuracy_real)
                    history_real['test_loss'].append(test_loss_real)
                    history_real['test_accuracy'].append(test_accuracy_real)
                    history_real['learning_rate'].append(scheduler_last_lr_real)
                    history_real['epoch_time'].append(epoch_time_real)

                    logger.info("--- Real Classifier --- :")
                    logger.info(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time_real:.2f}s - "
                        f"Train Loss: {train_loss_real:.4f} - Train Acc: {train_accuracy_real:.4f} - "
                        f"Test Loss: {test_loss_real:.4f} - Test Acc: {test_accuracy_real:.4f} - "
                        f"LR: {scheduler_last_lr_real:.6f}")
                    
                    if test_accuracy_real > best_accuracy_real:
                        best_accuracy_real = test_accuracy_real
                        best_model_real = copy.deepcopy(model_real)
                        no_improvement_epochs_real = 0
                    else:
                        no_improvement_epochs_real += 1
                        if no_improvement_epochs_real >= patience:
                            logger.info(f"Early stopping at epoch {epoch+1} for Real Classifier due to no improvement in test accuracy for {patience} consecutive epochs.\n")
                            stop_training_real = True        

                # Test the already trained real classifier
                elif use_also_real_classifier and skip_training_real and execute_only_one_epoch:
                    logger.info("\nSkipping the training loop for the real classifier as it is already trained.\n")
                    start_time_real = time.time()

                    test_loss_real, test_accuracy_real = test_epoch(
                        model=model_real,
                        test_dataloader=test_dataloader,
                        dataset_name = dataset_name,
                        criterion=criterion,
                        device=device
                    )

                    epoch_time_real = time.time() - start_time_real

                    logger.info("--- Pre-Trained Real Classifier Testing --- :")
                    logger.info(f"Epoch Time: {epoch_time_real:.2f}s - "
                    f"Test Loss: {test_loss_real:.4f} - Test Acc: {test_accuracy_real:.4f} - \n")

                    # Stop the epochs loop just for this classifier since it is already trained, while the synthetic classifier is not
                    execute_only_one_epoch = False


        # Test the both already trained synthetic and real classifiers
        else:
            
            logger.info("\nSkipping the training loop for the synthetic classifier as it is already trained.\n")
            start_time_synthetic = time.time()

            test_loss_synthetic, test_accuracy_synthetic = test_epoch(
                        model=model_synthetic,
                        test_dataloader=test_dataloader,
                        dataset_name = dataset_name,
                        criterion=criterion,
                        device=device
            )

            epoch_time_synthetic = time.time() - start_time_synthetic

            logger.info("--- Pre-Trained Synthetic Classifier Testing --- :")
            logger.info(f"Epoch Time: {epoch_time_synthetic:.2f}s - "
            f"Test Loss: {test_loss_synthetic:.4f} - Test Acc: {test_accuracy_synthetic:.4f} \n")


            if use_also_real_classifier:
                logger.info("Skipping the training loop for the real classifier as it is already trained.\n")
                start_time_real = time.time()

                test_loss_real, test_accuracy_real = test_epoch(
                        model=model_real,
                        test_dataloader=test_dataloader,
                        dataset_name = dataset_name,
                        criterion=criterion,
                        device=device
                )

                epoch_time_real = time.time() - start_time_real

                logger.info("--- Pre-Trained Real Classifier Testing --- :")
                logger.info(f"Epoch Time: {epoch_time_real:.2f}s - "
                f"Test Loss: {test_loss_real:.4f} - Test Acc: {test_accuracy_real:.4f} \n")


        # Record the end time of the pipeline
        end_time_pipeline = time.time()

        # Log the total time taken by the pipeline
        pipeline_time = end_time_pipeline - start_time_pipeline
        hours, remainder = divmod(pipeline_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        logger.info(f"\n Total time taken by the pipeline: {int(hours)}h {int(minutes)}m {int(seconds)}s \n")
        
        # RAM usage at the end of the pipeline
        logger.info(f"RAM usage at the end: {get_memory_usage():.2f} GB\n")

        # Clear the CUDA cache to avoid memory issues
        torch.cuda.empty_cache()


        if not skip_training_synthetic:

            # Save the synthetic classifier metadata
            metadata_synthetic = {
                'model': best_model_synthetic,
                'best_accuracy': best_accuracy_synthetic,
                'history': history_synthetic
            }

            # Log the best accuracy of the synthetic classifier
            logger.info(f"Accuracy of the Synthetic classifier: {metadata_synthetic['best_accuracy'] * 100:.2f} %\n") 

            # Save the synthetic classifier weights only if they don't already exist (they exist only if i'm running multiple times the same pipeline)
            synthetic_model_dir = "../storage/trained_synthetic_classifiers"
            os.makedirs(synthetic_model_dir, exist_ok=True)
            base_filename = f"{current_time}.pth"
            synthetic_model_path = os.path.join(synthetic_model_dir, base_filename)

            if not os.path.exists(synthetic_model_path):
                torch.save(best_model_synthetic.state_dict(), synthetic_model_path)
                logger.info(f"Saved synthetic classifier weights to {synthetic_model_path}\n")
            else:
                logger.info(f"Synthetic classifier weights not saved as {synthetic_model_path} already exists.\n")


        if use_also_real_classifier and not skip_training_real:

            # Save the real classifier metadata
            metadata_real = {
                'model': best_model_real,
                'best_accuracy': best_accuracy_real,
                'history': history_real
            }

            # Log the best accuracy of the real classifier
            logger.info(f"Accuracy of the Real classifier: {metadata_real['best_accuracy'] * 100:.2f} %\n") # * 100 to convert to percentage, and :.2f to round to 2 decimal places

            # Save the real classifier weights only if they don't already exist (they exist only if i'm running multiple times the same pipeline)
            real_model_dir = "../storage/trained_real_classifiers"
            os.makedirs(real_model_dir, exist_ok=True)
            base_filename = f"{current_time}.pth"
            real_model_path = os.path.join(real_model_dir, base_filename)

            if not os.path.exists(real_model_path):
                torch.save(best_model_real.state_dict(), real_model_path)
                logger.info(f"Saved real classifier weights to {real_model_path}\n")
            else:
                logger.info(f"Real classifier weights not saved as {real_model_path} already exists.\n")



    # Catch any error and log it
    except Exception as e:

        logger.error("\n An error occurred during the pipeline execution:", exc_info=True)
        raise e


    # Execute this even if an exception occurred
    finally:

        try:
            # Delete the temporary pickle file used in case the synthetic dataset was sampled from a pre-existing one
            if already_saved_synthetic_dataset_dir and os.path.exists(os.path.join("temp_pipeline_files", f"{current_time}_temp_samples_selected.pkl")):
                os.remove(os.path.join("temp_pipeline_files", f"{current_time}_temp_samples_selected.pkl"))
                logger.info("Deleted the temporary .pkl file used for sampling the synthetic dataset from a dataset on disk.\n")

            # Don't delete the synthetic dataset directory if it was sampled from a pre-existing one
            if (already_saved_synthetic_dataset_dir is None) and synthetic_dataset_dir and (os.path.exists(synthetic_dataset_dir)):
                # Delete the synthetic dataset if either save_synthetic_dataset is False, OR an exception occurred 
                if (not save_synthetic_dataset) or (sys.exc_info()[0] is not None):
                    logger.info(f"Deleting synthetic dataset at directory: {synthetic_dataset_dir}...\n")
                    shutil.rmtree(synthetic_dataset_dir)
                    logger.info(f"Deleted synthetic dataset.\n")
                else:
                    logger.info(f"Synthetic dataset saved at directory: {synthetic_dataset_dir}\n")
                    
            # Log the end of the pipeline
            logger.info("------------ End of Pipeline -----------------\n")

        except Exception as del_e:
            logger.error("An error occurred while deleting the temporary files/folders used during the script:", exc_info=True)