from torchvision import transforms

from utils import AdaptiveCenterCropAndResize


def build_train_transform(args):

    tfms = [
        AdaptiveCenterCropAndResize(
            (args.img_size, args.img_size)
        ),
    ]

    if args.aug_hflip > 0:
        tfms.append(
            transforms.RandomHorizontalFlip(
                p=args.aug_hflip
            )
        )

    if args.aug_vflip > 0:
        tfms.append(
            transforms.RandomVerticalFlip(
                p=args.aug_vflip
            )
        )

    if (
        args.aug_brightness > 0
        or args.aug_contrast > 0
        or args.aug_saturation > 0
        or args.aug_hue > 0
    ):
        tfms.append(
            transforms.ColorJitter(
                brightness=args.aug_brightness,
                contrast=args.aug_contrast,
                saturation=args.aug_saturation,
                hue=args.aug_hue,
            )
        )

    if args.aug_blur_prob > 0:
        tfms.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=args.aug_blur_kernel
                    )
                ],
                p=args.aug_blur_prob,
            )
        )

    if args.aug_random_erasing > 0:
        tfms.append(
            transforms.RandomErasing(
                p=args.aug_random_erasing,
            )
        )

    return transforms.Compose(tfms)


def build_test_transform(args):

    return transforms.Compose([
        AdaptiveCenterCropAndResize(
            (args.img_size, args.img_size)
        )
    ])