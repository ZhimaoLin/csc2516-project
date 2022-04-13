import pandas as pd


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

