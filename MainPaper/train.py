import pandas as pd
import cv2
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from pain_detector import PainDetector
from models.comparative_model import ConvNetOrdinalLateFusion
import matplotlib.pyplot as plt


DATA_SUMMARY_CSV_PATH = '../data_summary.csv'
DATA_SUMMARY_HEADER =  {"person":"person_name", "video":"video_name", "frame":"frame_number", "pspi":"pspi_score", "image":"image_path"}

TRAIN_DATA_CSV_PATH = 'train_data.csv'
TRAIN_DATA_CSV_HEADER = ("reference_image_path", "target_image_path", "pspi_score")

REFERENCE_IMAGE_SAMPLE_SIZE = 1
BATCH_SIZE = 10
NUM_OUTPUT = 7
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
    'image_size':160,
    'number_output':7,
    'image_sample_size':1,
    'batch_size': 10,  
    'drop_out': 0,
    'fc2_size': 200,
    'learning_rate': 0.001,
    'epoch': 5
}
ARGS.update(args_dict)
#endregion


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
        reference_image_path = self.df.iloc[idx][TRAIN_DATA_CSV_HEADER[0]]
        target_image_path = self.df.iloc[idx][TRAIN_DATA_CSV_HEADER[1]]
        reference_image = cv2.imread(reference_image_path)
        target_image = cv2.imread(target_image_path)

        reference_image_tensor = self.pain_detector.prep_image(reference_image)
        target_image_tensor = self.pain_detector.prep_image(target_image)

        input_tensor = torch.cat([reference_image_tensor, target_image_tensor], dim=1).squeeze(dim=0)

        pspi_score = self.df.iloc[idx, 2]
        output_tensor = torch.tensor(pspi_score) 

        return input_tensor, output_tensor
#endregion



# Write a row to a file. The row has to be a tuple of strings
# Input: "./data_summary.csv", ("person_name", "video_name", "frame_number", "pspi_score", "image_path"), "w"
# output: None
def write_row_to_file(file_path, row, mode="a"):
    with open(file_path, mode) as f:
        f.write(",".join(row) + "\n")



# Create a train data csv contains ("reference_image_path", "target_image_path", "pspi_score")
def create_data_csv(train_data_csv_path, data_summary_csv_path, ops):

    write_row_to_file(train_data_csv_path, TRAIN_DATA_CSV_HEADER, "w")
    df = pd.read_csv(data_summary_csv_path)  

    group_by_person_name = df.groupby([DATA_SUMMARY_HEADER["person"]])
    for name, group in group_by_person_name:
        target_group = group[group.pspi_score != 0]
        reference_group = group[group.pspi_score == 0]
        
        for index, row in target_group.iterrows():
            pspi_score = str(row[DATA_SUMMARY_HEADER["pspi"]])
            target_image_path = row[DATA_SUMMARY_HEADER["image"]]

            # Randomly pick reference images
            reference_sample_df = reference_group.sample(n=ops.image_sample_size, random_state=RANDOM_SEED)
            for i, s in reference_sample_df.iterrows():
                reference_image_path = s[DATA_SUMMARY_HEADER["image"]]
                data = (reference_image_path, target_image_path, pspi_score)
                write_row_to_file(train_data_csv_path, data)
    
    return train_data_csv_path


# def create_input_tensor(data_df, ops):
#     pain_detector = PainDetector(image_size=ops.image_size, checkpoint_path='checkpoints/59448122/59448122_3/model_epoch13.pt', num_outputs=ops.number_output)

#     input_list = []
#     for index, row in data_df.iterrows():
#         reference_image_path = row[TRAIN_DATA_CSV_HEADER[0]]
#         target_image_path = row[TRAIN_DATA_CSV_HEADER[1]]
#         reference_image = cv2.imread(reference_image_path)
#         target_image = cv2.imread(target_image_path)

#         reference_image_tensor = pain_detector.prep_image(reference_image)
#         target_image_tensor = pain_detector.prep_image(target_image)

#         frames = torch.cat([reference_image_tensor, target_image_tensor], dim=1)
#         input_list.append(frames)

#     input_tensor = torch.cat(input_list, dim=0)

#     return input_tensor


# def read_data_to_tensor(data_path, ops):
#     df = pd.read_csv(data_path)

#     train_df = df.sample(frac=0.8, random_state=RANDOM_SEED)
#     train_pspi_df = train_df[TRAIN_DATA_CSV_HEADER[2]]
    
#     test_df = df.drop(train_df.index)
#     test_pspi_df = test_df[TRAIN_DATA_CSV_HEADER[2]]

#     train_input_tensor = create_input_tensor(train_df, ops)
#     train_output_tensor = torch.tensor(train_pspi_df.values).reshape((-1, 1))

#     test_input_tensor = create_input_tensor(test_df, ops)
#     test_output_tensor = torch.tensor(test_pspi_df.values).reshape((-1, 1))

#     torch.save(train_input_tensor, 'train_input_tensor.pt')
#     torch.save(train_output_tensor, 'train_output_tensor.pt')
#     torch.save(test_input_tensor, 'test_input_tensor.pt')
#     torch.save(test_output_tensor, 'test_output_tensor.pt')

#     return train_input_tensor, train_output_tensor, test_input_tensor, test_output_tensor


# def create_dataloader(train_input_tensor, train_output_tensor, test_input_tensor, test_output_tensor, ops):
#     train_input_loader = torch.utils.data.DataLoader((train_input_tensor, train_output_tensor), batch_size=ops.batch_size)

#     for data in train_input_loader:
#         print(data)
#         break


#     train_output_loader = torch.utils.data.DataLoader(train_output_tensor, batch_size=ops.batch_size)
#     test_input_loader = torch.utils.data.DataLoader(test_input_tensor, batch_size=ops.batch_size)
#     test_output_loader = torch.utils.data.DataLoader(test_output_tensor, batch_size=ops.batch_size)

#     return train_input_loader, train_output_loader, test_input_loader, test_output_loader


def create_model(ops):
    model = ConvNetOrdinalLateFusion(num_outputs=ops.number_output, dropout=ops.drop_out, fc2_size=ops.fc2_size)
    if torch.cuda.is_available():
        model.cuda()
        print(f"Move model to GPU")

    print("---------------------------------------")
    print(model)
    print("---------------------------------------")

    return model


# def train(train_input_loader, train_output_loader, ops):
#     net = create_model(ops)
    
#     optimizer = optim.Adam(net.parameters(), lr=ops.learning_rate)

#     for epoch in range(ops.epoch):
#         for data in train_input_loader:
#             pass
    





def main():
    pain_detector = PainDetector(image_size=ARGS.image_size, checkpoint_path='checkpoints/59448122/59448122_3/model_epoch13.pt', num_outputs=ARGS.number_output)

    train_data_path = create_data_csv(TRAIN_DATA_CSV_PATH, DATA_SUMMARY_CSV_PATH, ARGS)
    dataset = MyDataset(TRAIN_DATA_CSV_PATH, pain_detector)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    for i_batch, sample_batched in enumerate(loader):
        print(i_batch) 
        print(sample_batched[0].size(), sample_batched[1].size())
        break

    

    # train_input_tensor, train_output_tensor, test_input_tensor, test_output_tensor = read_data_to_tensor(TRAIN_DATA_CSV_PATH, ARGS)

    # train_input_tensor = torch.load('train_input_tensor.pt')
    # train_output_tensor = torch.load('train_output_tensor.pt')
    # test_input_tensor = torch.load('test_input_tensor.pt')
    # test_output_tensor = torch.load('test_output_tensor.pt')

    # train_input_loader, train_output_loader, test_input_loader, test_output_loader = create_dataloader(train_input_tensor, train_output_tensor, test_input_tensor, test_output_tensor, ARGS)

    # test = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    # loader = torch.utils.data.DataLoader(test, batch_size=10)

    # for data in loader:
    #     print(data)
    


    # loader = torch.utils.data.DataLoader(train_input_tensor, batch_size=batch_size)
    # train_iter = iter(loader)

    # test = train_iter.next()





if __name__ == "__main__":
    main()
