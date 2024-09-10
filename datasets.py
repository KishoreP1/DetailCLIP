# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import json
import os
import pickle
import zipfile

import numpy as np
from PIL import Image, ImageFile

import torch
from torchvision import transforms
from torchvision import datasets as t_datasets
from torchvision.transforms import functional as F

import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def yfcc_loader(root, index):
    repo = index[:3]
    sub_repo = index[3: 6]
    file_img = index + '.jpg'
    # output15mv2 contains zip files
    path_zip = os.path.join(root, 'test_zip_3' , repo) + '.zip'
    with zipfile.ZipFile(path_zip, 'r') as myzip:
        url = os.path.join('data', 'images', repo, sub_repo, file_img)
        img = Image.open(myzip.open(url))
    return img.convert('RGB')


class ImageCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata):
        self.dataset = dataset
        self.root = root
        if self.dataset == 'yfcc15m':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset == 'coco':
            samples = defaultdict(list)
            with open(metadata) as f:
                annotations = json.load(f)['annotations']
            for ann in annotations:
                samples[ann['image_id']].append(ann['caption'])
            self.samples = [(k, v) for k, v in samples.items()]
        elif self.dataset == 'cc12m' or self.dataset == 'cc3m':
            self.samples = np.load(metadata, allow_pickle=True)
        elif self.dataset == 'redcaps':
            with open(metadata) as f:
                annotations = json.load(f)
            self.samples = [(ann['image_id'], ann['subreddit'], ann['caption']) for ann in annotations]
        elif self.dataset == 'mnist':
            mnist = t_datasets.MNIST(root, train=True, download=True)
            self.samples = [(image, f'a photo of the number: "{caption}".') for image, caption in mnist]
        
    def get_raw_item(self, i):
        if self.dataset == 'yfcc15m':
            index, hash, title, desc = self.samples[i]
            # caption = np.random.choice([title, desc])
            caption = title + ' ' + desc
            img = yfcc_loader(self.root, hash)
        elif self.dataset == 'coco':
            index, captions = self.samples[i]
            path = os.path.join(self.root, 'train2017', '{:012d}.jpg'.format(index))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'cc3m':
            ann = self.samples[i]
            filename, captions = ann['image_id'], ann['captions']
            path = os.path.join(self.root, str(filename))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'cc12m':
            ann = self.samples[i]
            filename, captions = ann['image_name'], ann['captions']
            path = os.path.join(self.root, filename)
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'redcaps':
            image_id, subreddit, caption = self.samples[i]
            path = os.path.join(self.root, subreddit, f"{image_id}.jpg")
            img = pil_loader(path)
        elif self.dataset == 'mnist':
            img, caption = self.samples[i]

        return img, caption


    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class ImageCaptionDatasetCLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return img, caption


class ImageCaptionDatasetSLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform, augment, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.augment = augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        image = self.transform(img)
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption, aug1, aug2
    
class  ImageCaptionDatasetACLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform, ema_transform, tokenizer=None):
        super().__init__(dataset, root, metadata)
        self.transform = transform
        self.ema_transform = ema_transform
        self.tokenizer = tokenizer
        self.get_three_crop = GetThreeRandomResizedCrop(224, scale=(0.5, 1.0))

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)
        res = self.get_three_crop(img)

        im1, ret1 = res[0]
        im2, ret2 = res[1]
        im3, ret3 = res[2]

        im1 = self.transform(im1)
        im2 = self.transform(im2)
        im3 = self.ema_transform(im3)

        pos = np.array([ret1,ret2,ret3])
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return [im1, im2, im3], pos, caption
    
class ImageCaptionDatasetDetailCLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        u = self.transform(img)
        v = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return [u, v], caption

class ImageCaptionDatasetSSL(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, augment):
        super().__init__(dataset, root, metadata)

        self.augment = augment

    def __getitem__(self, i):
        img, _ = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return aug1, aug2


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry['path']
    if entry['type'] == 'imagefolder':
        dataset = t_datasets.ImageFolder(os.path.join(root, entry['train'] if is_train else entry['test']),
            transform=transform)
    elif entry['type'] == 'special':
        if name == 'cifar10':
            dataset = t_datasets.CIFAR10(root, train=is_train,
                transform=transform, download=True)
        elif name == 'cifar100':
            dataset = t_datasets.CIFAR100(root, train=is_train,
                transform=transform, download=True)
        elif name == 'stl10':
            dataset = t_datasets.STL10(root, split='train' if is_train else 'test',
                transform=transform, download=True)
        elif name == 'mnist':
            dataset = t_datasets.MNIST(root, train=is_train,
                transform=transform, download=True)
    elif entry['type'] == 'filelist':
        path = entry['train'] if is_train else entry['test']
        val_images = os.path.join(root, path + '_images.npy')
        val_labels = os.path.join(root, path + '_labels.npy')
        if name == 'clevr_counts':
            target_transform = lambda x: ['count_10', 'count_3', 'count_4', 'count_5', 'count_6', 'count_7', 'count_8', 'count_9'].index(x)
        else:
            target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    else:
        raise Exception('Unknown dataset')

    return dataset

class GetThreeRandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            List[(cropped image, ret)] *3.
            The scale of the last image is larger than the first two.
        """
        ret1 = self.get_params(img, self.scale, self.ratio)
        ret2 = self.get_params(img, self.scale, self.ratio)

        try:
            _, height, width = F.get_dimensions(img)
        except:
            width, height = F.get_image_size(img)

        im1 = F.resized_crop(img, *ret1, self.size, self.interpolation)
        im2 = F.resized_crop(img, *ret2, self.size, self.interpolation)

        # zoom out
        ret3 = [0, 0, 0, 0]
        ret3[0], ret3[1], = min(ret1[0], ret2[0]), min(ret1[1], ret2[2])

        rh = max(ret1[0] + ret1[2], ret2[0] + ret2[2])
        rw = max(ret1[1] + ret1[3], ret2[1] + ret2[3])
        ret3[2], ret3[3] = rh - ret3[0], rw - ret3[1]

        ret3[0] = torch.randint(0, ret3[0] + 1, size=(1,)).item() if ret3[0] > 0 else ret3[0]
        ret3[1] = torch.randint(0, ret3[1] + 1, size=(1,)).item() if ret3[1] > 0 else ret3[1]

        ret3[2] = torch.randint(ret3[2], height - ret3[0] + 1, size=(1,)).item() if ret3[2] < height else ret3[2]
        ret3[3] = torch.randint(ret3[3], width - ret3[1] + 1, size=(1,)).item() if ret3[3] < width else ret3[3]

        im3 = F.resized_crop(img, *ret3, self.size, self.interpolation)

        return [(im1, ret1), (im2, ret2), (im3, ret3)]

def get_dataset(train_transform, tokenizer, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augment = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    if args.model.startswith('SIMCLR'):
        return ImageCaptionDatasetSSL(args.dataset, args.root, args.metadata, augment)
    elif args.model.startswith('CLIP'):
        return ImageCaptionDatasetCLIP(args.dataset, args.root, args.metadata, train_transform, tokenizer)
    elif args.model.startswith('SLIP'):
        return ImageCaptionDatasetSLIP(args.dataset, args.root, args.metadata, train_transform, augment, tokenizer)
    elif args.model.startswith('ACLIP'):
        ema_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
          ])
        return ImageCaptionDatasetACLIP(args.dataset, args.root, args.metadata, train_transform, ema_transform, tokenizer)
    elif args.model.startswith('DetailCLIP'):
        return ImageCaptionDatasetDetailCLIP(args.dataset, args.root, args.metadata, train_transform, tokenizer)



