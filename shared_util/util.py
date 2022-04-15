import torch
import torch.optim as optim
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import pearsonr


# Write a row to a file. The row has to be a tuple of strings
# Input: "./data_summary.csv", ("person_name", "video_name", "frame_number", "pspi_score", "image_path"), "w"
# output: None
def write_row_to_file(file_path, row, mode="a"):
    with open(file_path, mode) as f:
        f.write(",".join(row) + "\n")


def split_dataset(data_summary_path, train_data_csv_path, test_data_csv_path, random_state, train_fraction=0.8):
    df = pd.read_csv(data_summary_path)

    train_df = df.sample(frac=train_fraction, random_state=random_state)
    test_df = df.drop(train_df.index)

    train_df.to_csv(train_data_csv_path, index=False)
    test_df.to_csv(test_data_csv_path, index=False)



def draw_line_chart(x_list, y_list, title, x_label, y_label, save_path, is_show=False):
    plt.figure()
    plt.plot(x_list, y_list)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    plt.tight_layout()
    if is_show:
        plt.show()
    plt.savefig(save_path)
    plt.close()  


def save_trained_model(model, path):
    torch.save(model.state_dict(), path)

def load_trained_model(path, create_model_function, ops):
    trained_model = create_model_function(ops)
    trained_model.load_state_dict(torch.load(path))
    trained_model.eval()
    return trained_model


def sample_data(data_path, n, random_state):
    df = pd.read_csv(data_path)
    train_df = df.sample(n=n, random_state=random_state)
    train_df.to_csv(data_path, index=False)


def calculate_accuracy_score(y_true, y_pred):
    y_true = np.array(y_true).round()
    y_pred = np.array(y_pred).round()
    result = accuracy_score(y_true, y_pred)
    return result

def calculate_pearson_correlation(x, y):
    result = pearsonr(x, y)
    return result

def calculate_f1_score(y_true, y_pred, threshold):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true = y_true > threshold
    y_pred = y_pred > threshold

    result = f1_score(y_true, y_pred)
    return result

