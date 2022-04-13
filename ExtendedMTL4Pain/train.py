import sys 
import os
sys.path.append(os.path.abspath("../shared_util"))
from util import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
from MyDataset import *
import matplotlib.pyplot as plt



DATA_SUMMARY_CSV_PATH = '../data_summary.csv'
TRAIN_DATA_CSV_PATH = 'train_data.csv'
TEST_DATA_CSV_PATH = 'test_data.csv'
TRAIN_FRACTION = 0.8
RANDOM_SEED = 1


#region arguments section
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


ARGS = AttrDict()
args_dict = {
    'image_scale_to_before_crop': 256,
    'image_size': 160,
    'number_output':1,
    'batch_size': 50,  
    'overall_learning_rate': 0.00001,
    'last_layer_learning_rate': 0.0001,
    'weight_decay': 0.0005,
    'epoch': 1


}
ARGS.update(args_dict)
#endregion



def create_model(ops):
    """ VGG16_bn """
    model_ft = models.vgg16_bn(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, ops.number_output)

    if torch.cuda.is_available():
        model_ft.cuda()
        print(f"Move model to GPU")

    print("---------------------------------------")
    print(model_ft)
    print("---------------------------------------")
    return model_ft

    

def train(train_data_path, ops):
    net = create_model(ops)

    # Extract parameters
    last_layer = list(net.children())[-1]
    try:
        last_layer = last_layer[-1]
    except:
        last_layer = last_layer
    ignored_params = list(map(id, last_layer.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                    net.parameters())

    # optimizer_ft = optim.Adam([
    #             {'params': base_params},
    #             {'params': last_layer.parameters(), 'lr': 1e-4}
    #         ], lr=1e-5, weight_decay=5e-4)

    optimizer_ft = optim.Adam([
                {'params': base_params},
                {'params': last_layer.parameters(), 'lr': ops.last_layer_learning_rate}
            ], lr=ops.overall_learning_rate, weight_decay=ops.weight_decay)


    criterion = nn.MSELoss()


    transform_function = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(ops.image_scale_to_before_crop), # 256
        transforms.CenterCrop(ops.image_size), # 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
    ])

    train_data = MyDataset(train_data_path, ops, transform=transform_function)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=ops.batch_size, shuffle=True)

    for epoch in range(ops.epoch):
        print(f"Epoch {epoch+1}/{ops.epoch}")
        print('-' * 10)

        for i_batch, sample_batched in enumerate(train_data_loader):
            net.zero_grad()
            with torch.set_grad_enabled(True):
                X = sample_batched[0]
                y = sample_batched[1]

                if torch.cuda.is_available():
                    X = X.cuda().float()
                    y = y.cuda().view((-1, 1)).float()

                output = net(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer_ft.step()

            print(f"Batch index [{i_batch}], Train loss is [{loss}]")





def main():
    split_dataset(DATA_SUMMARY_CSV_PATH, TRAIN_DATA_CSV_PATH, TEST_DATA_CSV_PATH, RANDOM_SEED, train_fraction=TRAIN_FRACTION)
    train(TRAIN_DATA_CSV_PATH, ARGS)



if __name__ == '__main__':
    main()

