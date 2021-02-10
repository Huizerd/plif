import torchvision.transforms as transform_lib


def zeromean_unitvar_transform():
    transforms = transform_lib.Compose(
        [
            transform_lib.ToTensor(),
            transform_lib.Normalize(0.0, 1.0),
        ]
    )
    return transforms
