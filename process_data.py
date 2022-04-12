import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# global variables
PSPI_DIR = 'data\Frame_Labels\Frame_Labels\PSPI'
IMAGES_DIR = 'data\Images\Images'
DATA_SUMMARY_CSV = 'data_summary.csv'



# Write a row to a file. The row has to be a tuple of strings
# Input: "./data_summary.csv", ("person_name", "video_name", "frame_number", "pspi_score", "image_path"), "w"
# output: None
def write_row_to_file(file_path, row, mode="a"):
    with open(file_path, mode) as f:
        f.write(",".join(row) + "\n")


# Create a data summary csv file
# Input: 'data\Frame_Labels\Frame_Labels\PSPI', 'data\Images\Images', 'data_summary.csv'
# output: The path to the csv file
def create_data_summary_csv(pspi_dir, images_dir, data_summary_csv):
    # The header of the csv
    header =  ("person_name", "video_name", "frame_number", "pspi_score", "image_path")
    write_row_to_file(data_summary_csv, header, "w")

    for root, dirs, files in os.walk(pspi_dir):
        for file in files:
            # Process labels
            label_path = os.path.join(root, file)
            label_path = label_path.replace("/", "\\")

            path_array = label_path.split("\\")
            person_name = path_array[4]
            video_name = path_array[5]
            
            frame_number = '0'
            m = re.search(video_name + '(\d+)', path_array[6])
            if m:
                frame_number = m.group(1)

            pspi_score = 0.0
            with open(label_path) as f:
                contents = f.read()
                contents = str(float(contents))
                pspi_score = contents

            # Construct image paths
            image_file_name = video_name + frame_number + ".png"
            image_path = os.path.join(images_dir, person_name, video_name, image_file_name)
            image_path = os.path.abspath(image_path)

            # Write to the csv 
            row = (person_name, video_name, frame_number, pspi_score, image_path)
            write_row_to_file(data_summary_csv, row)

    return data_summary_csv




def main():
    data_summary_csv_path = create_data_summary_csv(PSPI_DIR, IMAGES_DIR, DATA_SUMMARY_CSV)



if __name__ == "__main__":
    main()

