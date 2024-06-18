import os
import numpy as np

from torchvision import datasets
from datasets.customDataset import Custom_Dataset


def init_datasets(args, transform_train, transform_test):
    print("employ dataset {}".format(args.dataset))

    if args.dataset == "cifar10":
        train_datasets = datasets.CIFAR10(root=os.path.expanduser('/data1'), train=True, download=True, transform=transform_train)
        val_datasets = datasets.CIFAR10(root=os.path.expanduser('/data1'), train=False, download=True, transform=transform_test)
        test_datasets = datasets.CIFAR10(root=os.path.expanduser('/data1'), train=False, download=True, transform=transform_test)
        num_classes = len(np.unique(train_datasets.targets))

    elif args.dataset == "cifar100":
        train_datasets = datasets.CIFAR100(root=os.path.expanduser('/data1'), train=True, download=True, transform=transform_train)
        val_datasets = datasets.CIFAR100(root=os.path.expanduser('/data1'), train=False, download=True, transform=transform_test)
        test_datasets = datasets.CIFAR100(root=os.path.expanduser('/data1'), train=False, download=True, transform=transform_test)
        num_classes = len(np.unique(train_datasets.targets))

    elif args.dataset == 'Pet37':
        train_datasets = datasets.OxfordIIITPet(root='/data1', download=True, split='trainval', transform=transform_train)
        val_datasets = datasets.OxfordIIITPet(root='/data1', download=True, split='test', transform=transform_test)
        test_datasets = datasets.OxfordIIITPet(root='/data1', download=True, split='test', transform=transform_test)
        num_classes = len(np.unique(train_datasets.classes))

        # root, txt_train, txt_val, txt_test = get_pet37_data_split(args.data, args.customSplit)

        # train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
        # val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
        # test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)
    elif args.dataset == 'food101':
        train_datasets = datasets.Food101(root='/data1', download=True, split='train', transform=transform_train)
        val_datasets = datasets.Food101(root='/data1', download=True, split='test', transform=transform_test)
        test_datasets = datasets.Food101(root='/data1', download=True, split='test', transform=transform_test)
        num_classes = len(np.unique(train_datasets.classes))

        # root, txt_train, txt_val, txt_test = get_food101_data_split(args.data, args.customSplit)

        # train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
        # val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
        # test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)
    elif args.dataset == 'flowers':
        train_datasets = datasets.Flowers102(root='/data1', download=True, split='train', transform=transform_train)
        val_datasets = datasets.Flowers102(root='/data1', download=True, split='val', transform=transform_test)
        test_datasets = datasets.Flowers102(root='/data1', download=True, split='test', transform=transform_test)
        num_classes = len(np.unique(train_datasets._labels))

        # root, txt_train, txt_val, txt_test = get_flowers_data_split(args.data, args.customSplit)

        # train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
        # val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
        # test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)
    elif args.dataset == 'stanford_car':
        #test_datasets = datasets.StanfordCars(root='/home/kzheng44/data1', download=True, split='test', transform=transform_test)
        train_datasets = datasets.StanfordCars(root='/home/kzheng44/data1', download=True, split='train', transform=transform_train)
        test_datasets = datasets.StanfordCars(root='/home/kzheng44/data1', download=True, split='test', transform=transform_test)
        val_datasets = datasets.StanfordCars(root='/home/kzheng44/data1', download=True, split='test', transform=transform_test)
        num_classes = len(np.unique(train_datasets.classes))

        # root, txt_train, txt_val, txt_test = get_stanford_car_data_split(args.data, args.customSplit)

        # train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
        # val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
        # test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)
    elif args.dataset == 'aircraft':
        train_datasets = datasets.FGVCAircraft(root='/data1', download=True, split='train', transform=transform_train)
        val_datasets = datasets.FGVCAircraft(root='/data1', download=True, split='val', transform=transform_test)
        test_datasets = datasets.FGVCAircraft(root='/data1', download=True, split='test', transform=transform_test)
        num_classes = len(np.unique(train_datasets._labels))

        # root, txt_train, txt_val, txt_test = get_aircraft_data_split(args.data, args.customSplit)

        # train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
        # val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
        # test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)
    elif args.dataset == 'cub200':
        root, txt_train, txt_val, txt_test = get_cub200_data_split(args.data, args.customSplit)

        train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
        val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
        test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)
    else:
        raise ValueError("Dataset of {} is not found".format(args.dataset))

    return train_datasets, val_datasets, test_datasets, num_classes


def get_pet37_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir(os.path.expanduser('~/dataset/pets')):
            root = os.path.expanduser('~/dataset/pets')
        elif os.path.isdir('./dataset/pets'):
            root = './dataset/pets'
        else:
            assert False

    return root


def get_pet37_data_split(root, customSplit, ssl=False):
    root = get_pet37_path(root)

    txt_train = "split/Pet37/Pet37_train.txt"
    txt_val = "split/Pet37/Pet37_val.txt"
    txt_test = "split/Pet37/Pet37_test.txt"

    if customSplit != '':
        txt_train = "split/Pet37/{}.txt".format(customSplit)

    if ssl:
        assert customSplit == ''
        train_idx = "split/Pet37/Pet37_trainval.txt"
        return root, train_idx, None, None

    return root, txt_train, txt_val, txt_test


def get_food101_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir(os.path.expanduser('~/dataset/food101')):
            root = os.path.expanduser('~/dataset/food101')
        elif os.path.isdir('./dataset/food101'):
            root = './dataset/food101'
        else:
            assert False

    return root


def get_food101_data_split(root, customSplit):
    root = get_food101_path(root)

    txt_train = "split/food101/train.txt"
    txt_val = "split/food101/val.txt"
    txt_test = "split/food101/val.txt"

    if customSplit != '':
        txt_train = "split/food101/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_flowers_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir(os.path.expanduser('~/dataset/flowers102')):
            root = os.path.expanduser('~/dataset/flowers102')
        elif os.path.isdir('./dataset/flowers102'):
            root = './dataset/flowers102'
        else:
            assert False

    return root


def get_flowers_data_split(root, customSplit):
    root = get_flowers_path(root)

    txt_train = "split/flowers/train.txt"
    txt_val = "split/flowers/val.txt"
    txt_test = "split/flowers/val.txt"

    if customSplit != '':
        txt_train = "split/flowers/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_stanford_car_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir(os.path.expanduser('~/dataset/stanford_car')):
            root = os.path.expanduser('~/dataset/stanford_car')
        elif os.path.isdir('./dataset/stanford_car'):
            root = './dataset/stanford_car'
        else:
            assert False

    return root


def get_stanford_car_data_split(root, customSplit):
    root = get_stanford_car_path(root)

    txt_train = "split/stanford_car/train.txt"
    txt_val = "split/stanford_car/val.txt"
    txt_test = "split/stanford_car/val.txt"

    if customSplit != '':
        txt_train = "split/stanford_car/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_aircraft_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir(os.path.expanduser('~/dataset/aircraft')):
            root = os.path.expanduser('~/dataset/aircraft')
        elif os.path.isdir('./dataset/aircraft'):
            root = './dataset/aircraft'
        else:
            assert False

    return root


def get_aircraft_data_split(root, customSplit):
    root = get_aircraft_path(root)

    txt_train = "split/aircraft/train.txt"
    txt_val = "split/aircraft/val.txt"
    txt_test = "split/aircraft/val.txt"

    if customSplit != '':
        txt_train = "split/aircraft/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_cub200_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir(os.path.expanduser('~/dataset/cub200')):
            root = os.path.expanduser('~/dataset/cub200')
        elif os.path.isdir('./dataset/cub200'):
            root = './dataset/cub200'
        else:
            assert False

    return root


def get_cub200_data_split(root, customSplit):
    root = get_cub200_path(root)

    txt_train = "split/cub200/train.txt"
    txt_val = "split/cub200/val.txt"
    txt_test = "split/cub200/val.txt"

    if customSplit != '':
        txt_train = "split/cub200/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test
