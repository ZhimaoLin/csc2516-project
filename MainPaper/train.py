import torch
import torch.optim as optim
from torch import nn
import pandas as pd
import numpy as np
from MyDataset import MyDataset
from models.comparative_model import ConvNetOrdinalLateFusion
import sys 
import os
sys.path.append(os.path.abspath("../shared_util"))
from util import *
import matplotlib.pyplot as plt


DATA_SUMMARY_CSV_PATH = '../data_summary.csv'
DATA_SUMMARY_HEADER =  {"person":"person_name", "video":"video_name", "frame":"frame_number", "pspi":"pspi_score", "image":"image_path"}

DATA_CSV_PATH = 'data.csv'
DATA_CSV_HEADER = ("reference_image_path", "target_image_path", "pspi_score")
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
    'image_scale_to_before_crop': 320,
    'image_size': 160,
    'number_output': 1,
    'image_sample_size': 1,
    'batch_size': 200,  
    'drop_out': 0,
    'fc2_size': 200,
    'learning_rate': 0.001,
    'epoch': 1
}
ARGS.update(args_dict)
#endregion

'''
#region dataset
class MyDataset(torch.utils.data.Dataset):
    """Construct my own dataset"""

    def __init__(self, path_to_csv, pain_detector, transform=None):
        """
        Args:
            path_to_csv (string): Path to the csv file
            pain_detector (PainDetector): An object of PainDetector
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(path_to_csv)
        self.pain_detector = pain_detector
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        reference_image_path = self.df.iloc[idx][DATA_CSV_HEADER[0]]
        target_image_path = self.df.iloc[idx][DATA_CSV_HEADER[1]]
        reference_image = cv2.imread(reference_image_path)
        target_image = cv2.imread(target_image_path)

        reference_image_tensor = self.pain_detector.prep_image(reference_image)
        target_image_tensor = self.pain_detector.prep_image(target_image)

        input_tensor = torch.cat([reference_image_tensor, target_image_tensor], dim=1).squeeze(dim=0)

        pspi_score = self.df.iloc[idx, 2]
        output_tensor = torch.tensor(pspi_score) 

        # input_tensor.requires_grad = True
        # output_tensor.requires_grad = True

        return input_tensor, output_tensor
#endregion
'''


# Write a row to a file. The row has to be a tuple of strings
# Input: "./data_summary.csv", ("person_name", "video_name", "frame_number", "pspi_score", "image_path"), "w"
# output: None
# def write_row_to_file(file_path, row, mode="a"):
#     with open(file_path, mode) as f:
#         f.write(",".join(row) + "\n")



# Create a train data csv contains ("reference_image_path", "target_image_path", "pspi_score")
def create_data_csv(data_csv_path, data_summary_csv_path, ops):

    write_row_to_file(data_csv_path, DATA_CSV_HEADER, "w")
    df = pd.read_csv(data_summary_csv_path)  

    group_by_person_name = df.groupby([DATA_SUMMARY_HEADER["person"]])
    for name, group in group_by_person_name:
        target_group = group[group.pspi_score != 0]
        reference_group = group[group.pspi_score == 0]
        
        for index, row in target_group.iterrows():
            pspi_score = str(row[DATA_SUMMARY_HEADER["pspi"]])
            target_image_path = row[DATA_SUMMARY_HEADER["image"]]
                
            # Randomly pick reference images
            reference_sample_df = reference_group.sample(n=ops.image_sample_size)
            for i, s in reference_sample_df.iterrows():
                reference_image_path = s[DATA_SUMMARY_HEADER["image"]]
                data = (reference_image_path, target_image_path, pspi_score)
                write_row_to_file(data_csv_path, data)

    # Split the data into train set and test set
    df = pd.read_csv(data_csv_path)
    train_df = df.sample(frac=0.8, random_state=RANDOM_SEED)
    test_df = df.drop(train_df.index)

    train_df.to_csv(TRAIN_DATA_CSV_PATH, index=False)
    test_df.to_csv(TEST_DATA_CSV_PATH, index=False)
    
    # return TRAIN_DATA_CSV_PATH, TEST_DATA_CSV_PATH


def create_data_loader(data_csv_path, pain_detector, is_shuffle, ops):
    dataset = MyDataset(data_csv_path, pain_detector)
    loader = torch.utils.data.DataLoader(dataset, batch_size=ops.batch_size, shuffle=is_shuffle)
    return loader



def create_model(ops):
    model = ConvNetOrdinalLateFusion(num_outputs=ops.number_output, dropout=ops.drop_out, fc2_size=ops.fc2_size)
    if torch.cuda.is_available():
        model.cuda()
        print(f"Move model to GPU")

    print("---------------------------------------")
    print(model)
    print("---------------------------------------")

    return model



def train(train_data_path, ops):
    train_dataset = MyDataset(train_data_path, ops)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=ops.batch_size, shuffle=True)

    net = create_model(ops)
    
    optimizer = optim.Adam(net.parameters(), lr=ops.learning_rate)
    mse_loss = nn.MSELoss()

    train_loss_for_each_epoch_list = []

    for epoch in range(ops.epoch):
        print(f"Epoch {epoch+1}/{ops.epoch}")
        print('-' * 10)
        train_loss_for_each_batch_list = []
        for i_batch, sample_batched in enumerate(train_data_loader):
            net.zero_grad()
            with torch.set_grad_enabled(True):
                X, y = sample_batched[0], sample_batched[1]

                if torch.cuda.is_available():
                    X = X.cuda().float()
                    y = y.cuda().view((-1, 1)).float()
                    # print(f"Move data to GPU")
                
                output = net(X, return_features=False)
                loss = mse_loss(output, y)
                loss.backward()
                optimizer.step()

                train_loss_for_each_batch_list.append(loss.item())
                print(f"Batch index [{i_batch}], Train loss is [{loss.item()}]")

        
        x_list = range(len(train_loss_for_each_batch_list))
        title = f"Epoch={epoch}, Train loss of each batch"
        x_label = "Batch number"
        y_label = "Loss"
        save_path = os.path.join("./result", f"Epoch_{epoch}_loss.png")
        draw_line_chart(x_list, train_loss_for_each_batch_list, title, x_label, y_label, save_path)


        avg_loss_for_each_epoch = np.array(train_loss_for_each_batch_list).mean()
        train_loss_for_each_epoch_list.append(avg_loss_for_each_epoch)


    x_list = range(len(train_loss_for_each_epoch_list))
    title = "Train loss of each epoch"
    x_label = "Epoch number"
    y_label = "Loss"
    save_path = os.path.join("./result", "Train_loss.png")
    draw_line_chart(x_list, train_loss_for_each_epoch_list, title, x_label, y_label, save_path)

    return net




def evaluation(net, test_data_loader):
    mse_loss = nn.MSELoss()

    total_number_batch = 0
    total_loss = 0
    for i_batch, sample_batched in enumerate(test_data_loader):
        
        with torch.set_grad_enabled(False):
            X, y = sample_batched[0], sample_batched[1]

            if torch.cuda.is_available():
                X = X.cuda().float()
                y = y.cuda().view((-1, 1)).float()
                # print(f"Move data to GPU")

            output = net(X, return_features=False)
            loss = mse_loss(output, y)

            total_loss += loss

        total_number_batch += 1

    avg_loss = total_loss / total_number_batch

    print(f"Average loss is [{avg_loss}]")



def main():
    # pain_detector = PainDetector(image_size=ARGS.image_size, checkpoint_path='checkpoints/59448122/59448122_3/model_epoch13.pt', num_outputs=ARGS.number_output)

    create_data_csv(DATA_CSV_PATH, DATA_SUMMARY_CSV_PATH, ARGS)
    split_dataset(DATA_CSV_PATH, TRAIN_DATA_CSV_PATH, TEST_DATA_CSV_PATH, RANDOM_SEED, TRAIN_FRACTION)

    # MyDataset(TRAIN_DATA_CSV_PATH, ARGS)



    # train_data_loader = create_data_loader(TRAIN_DATA_CSV_PATH, pain_detector, True, ARGS)
    # test_data_loader = create_data_loader(TEST_DATA_CSV_PATH, pain_detector, True, ARGS)

    net = train(TRAIN_DATA_CSV_PATH, ARGS)
    evaluation(net, TEST_DATA_CSV_PATH)



if __name__ == "__main__":
    main()
