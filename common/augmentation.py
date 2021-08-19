import albumentations as albu


def training_augmentation(size=(512, 512)):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=10, shift_limit=0.1, p=0.8, border_mode=0),
        albu.IAAAdditiveGaussianNoise(p=0.1),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
    ]
    if size is not None:
        train_transform.extend([
            albu.PadIfNeeded(min_height=size[0], min_width=size[1], border_mode=0, always_apply=True),
            albu.RandomCrop(height=size[0], width=size[1], always_apply=True),
        ])
    return albu.Compose(train_transform, p=0.5)


def base_augmentation(size=(512, 512)):
    test_transform = [
        albu.PadIfNeeded(size[0], size[1], border_mode=0, always_apply=True),
        albu.RandomCrop(height=size[0], width=size[1], always_apply=True)
    ]
    return albu.Compose(test_transform)




