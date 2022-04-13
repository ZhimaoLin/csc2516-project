import sys 
import os
sys.path.append(os.path.abspath("./shared_util"))
# sys.path.append(os.path.abspath("../shared_util/"))
import re

from shared_util.preprocess import PreProcess
from shared_util.util import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from pain_detector import PainDetector

# global variables
PSPI_DIR = './test/Frame_Labels/Frame_Labels/PSPI'
IMAGES_DIR = './test/Images/Images'
DATA_SUMMARY_CSV = 'data_summary.csv'


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
    'image_size': 160
}
ARGS.update(args_dict)
#endregion



# Create a data summary csv file
# Input: 'data\Frame_Labels\Frame_Labels\PSPI', 'data\Images\Images', 'data_summary.csv'
# output: The path to the csv file
def create_data_summary_csv(pspi_dir, images_dir, data_summary_csv):
    preprocess = PreProcess(ARGS)

    # The header of the csv
    header =  ("person_name", "video_name", "frame_number", "pspi_score", "image_path")
    write_row_to_file(data_summary_csv, header, "w")

    for root, dirs, files in os.walk(pspi_dir):
        for file in files:
            # Process labels
            label_path = os.path.join(root, file)
            label_path = label_path.replace("\\", "/")

            path_array = label_path.split("/")
            person_name = path_array[5]
            video_name = path_array[6]
            
            frame_number = '0'
            m = re.search(video_name + '(\d+)', path_array[7])
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

            if preprocess.test_image(image_path):
                # Write to the csv 
                row = (person_name, video_name, frame_number, pspi_score, image_path)
                write_row_to_file(data_summary_csv, row)
            else:
                print(f"Skip image: [{image_path}]")

    return data_summary_csv



def main():
    data_summary_csv_path = create_data_summary_csv(PSPI_DIR, IMAGES_DIR, DATA_SUMMARY_CSV)



if __name__ == "__main__":
    main()

