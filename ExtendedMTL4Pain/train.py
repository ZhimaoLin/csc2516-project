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

import time



DATA_SUMMARY_CSV_PATH = '../data_summary.csv'
TRAIN_DATA_CSV_PATH = 'train_data.csv'
TEST_DATA_CSV_PATH = 'test_data.csv'

RESULT_PATH = "result"

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
        # if opts.__dict__[key]:
        print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


ARGS = AttrDict()
args_dict = {
    'image_scale_to_before_crop': 200, # 300 is better old = 256
    'image_size': 160,
    'number_output':1,
    'batch_size': 50,  
    'overall_learning_rate': 0.0001,
    'last_layer_learning_rate': 0.001,
    'weight_decay': 0.0005,
    'epoch': 5,
    # For testing the code
    'train_sample': 150,
    'test_sample': 50
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

    train_loss_for_each_epoch_list = []

    for epoch in range(ops.epoch):
        print(f"Epoch {epoch+1}/{ops.epoch}")
        print('-' * 10)
        train_loss_for_each_batch_list = []

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

            train_loss_for_each_batch_list.append(loss.item())
            print(f"Batch index [{i_batch}], Train loss is [{loss}]")

        x_list = range(len(train_loss_for_each_batch_list))
        title = f"Epoch={epoch}, Train loss of each batch"
        x_label = "Batch number"
        y_label = "Loss"
        save_path = os.path.join(RESULT_PATH, f"Epoch_{epoch+1}_loss.png")
        draw_line_chart(x_list, train_loss_for_each_batch_list, title, x_label, y_label, save_path)

        avg_loss_for_each_epoch = np.array(train_loss_for_each_batch_list).mean()
        train_loss_for_each_epoch_list.append(avg_loss_for_each_epoch)


    x_list = range(len(train_loss_for_each_epoch_list))
    title = "Train loss of each epoch"
    x_label = "Epoch number"
    y_label = "Loss"
    save_path = os.path.join(RESULT_PATH, "Train_loss.png")
    draw_line_chart(x_list, train_loss_for_each_epoch_list, title, x_label, y_label, save_path)

    return net



def evaluation(net, test_data_path, ops):
    transform_function = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(ops.image_scale_to_before_crop), # 256
        transforms.CenterCrop(ops.image_size), # 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
    ])

    test_dataset = MyDataset(test_data_path, ops, transform=transform_function)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=ops.batch_size, shuffle=True)

    mse_loss = nn.MSELoss()

    loss_list = []
    prediction_list = []
    truth_list = []

    for i_batch, sample_batched in enumerate(test_data_loader):
        
        with torch.set_grad_enabled(False):
            X, y = sample_batched[0], sample_batched[1]

            if torch.cuda.is_available():
                X = X.cuda().float()
                y = y.cuda().view((-1, 1)).float()
                # print(f"Move data to GPU")

            output = net(X)
            loss = mse_loss(output, y)

        loss_list.append(loss.item())
        prediction_list += output.squeeze().tolist()
        truth_list += y.squeeze().tolist()

    avg_loss = np.array(loss_list).mean()
    accuracy = calculate_accuracy_score(truth_list, prediction_list)
    correlation = calculate_pearson_correlation(prediction_list, truth_list)
    f1 = calculate_f1_score(truth_list, prediction_list, 2)

    print(f"Average loss is [{avg_loss}]")
    print(f"Accuracy score is [{accuracy}]")
    print(f"Pearson correlation score is [{correlation}]")
    print(f"F1 score is [{f1}]")



def main():
    split_dataset(DATA_SUMMARY_CSV_PATH, TRAIN_DATA_CSV_PATH, TEST_DATA_CSV_PATH, RANDOM_SEED, train_fraction=TRAIN_FRACTION)

    # region Test Code
    # sample_data(TRAIN_DATA_CSV_PATH, ARGS.train_sample, RANDOM_SEED)
    # endregion

    # for batch_size in [50, 100, 150, 200]:
    #     ARGS.batch_size = batch_size
    #     global RESULT_PATH
    #     RESULT_PATH = "result"
    #     RESULT_PATH = os.path.join(RESULT_PATH, f"batch_size_{batch_size}")

    if not os.path.isdir(RESULT_PATH):
        os.mkdir(RESULT_PATH)

    print_opts(ARGS)

    start = time.time()
    net = train(TRAIN_DATA_CSV_PATH, ARGS)
    end = time.time()
    print(f"Runtime of the program is [{(end - start)/60}] minutes")

    model_path = os.path.join(RESULT_PATH, "model.pt")
    save_trained_model(net, model_path)

    # old_model = load_trained_model(model_path, create_model, ARGS)

    # region Test Code
    # sample_data(TEST_DATA_CSV_PATH, ARGS.test_sample, RANDOM_SEED)
    # endregion

    evaluation(net, TEST_DATA_CSV_PATH, ARGS) 



if __name__ == '__main__':
    main()

