import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

import oxford_flowers_dataset, oxford_pets_dataset

def build_dataset(is_train, args, transform_train=None):
    if not transform_train:
        transform_train = is_train
    transform = build_transform(transform_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "flowers":
        dataset = oxford_flowers_dataset.Flowers(root=args.data_path, 
                                     train=is_train,
                                     download=False,
                                     transform=transform)
        nb_classes = 102
    elif args.data_set == "pets":
        dataset = oxford_pets_dataset.Pets(root=args.data_path,
                                     train=is_train,
                                     download=True,
                                     transform=transform)
        nb_classes = 37
    elif args.data_set == "stl10":
        if is_train:
            dataset = datasets.STL10(root=args.data_path,
                                         split='train',
                                         download=True,
                                         transform=transform)
        else:
            dataset = datasets.STL10(root=args.data_path,
                                     split='test',
                                     download=True,
                                     transform=transform)
        nb_classes = 10
    elif args.data_set == "food101":
        if is_train:
            dataset = datasets.Food101(root=args.data_path,
                                         split='train',
                                         download=True,
                                         transform=transform)
        else:
            dataset = datasets.Food101(root=args.data_path,
                                     split='test',
                                     download=True,
                                     transform=transform)
        nb_classes = 101
    else:
        raise NotImplementedError()
    args.nb_classes = nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
