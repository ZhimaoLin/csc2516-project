# csc2516-project

The purpose of this project is to compare the following papers:

- Unobtrusive Pain Monitoring in Older Adults With Dementia Using Pairwise and Contrastive Training (**main paper**)
  - Paper link: [https://ieeexplore.ieee.org/document/9298886](https://ieeexplore.ieee.org/document/9298886)
  - GitHub repo: [https://github.com/TaatiTeam/pain_detection_demo.git](https://github.com/TaatiTeam/pain_detection_demo.git)
  - PDF: [https://tspace.library.utoronto.ca/bitstream/1807/104048/1/Dementia_Pain_Detection_TSpace.pdf](https://tspace.library.utoronto.ca/bitstream/1807/104048/1/Dementia_Pain_Detection_TSpace.pdf)
- Unobtrusive Pain Monitoring in Older Adults With Dementia Using Pairwise and Contrastive Training (**ExtendedMTL4Pain**)
  - Paper link: [https://proceedings.mlr.press/v116/xu20a.html](https://proceedings.mlr.press/v116/xu20a.html)
  - GitHub repo: [https://github.com/xiaojngxu/ExtendedMTL4Pain.git](https://github.com/xiaojngxu/ExtendedMTL4Pain.git)
  - PDF: [http://proceedings.mlr.press/v116/xu20a/xu20a.pdf](http://proceedings.mlr.press/v116/xu20a/xu20a.pdf)

## Data 

We use a new data set to compare those models in the paper. The data can be downloaded from [here](https://www.kaggle.com/datasets/coder98/emotionpain).

You need to download the `archive.zip` file from [Kaggle](https://www.kaggle.com/datasets/coder98/emotionpain) to the `csc2516-project/data` folder. Then, extract the zip file to the `data` folder. The structure of the `data` folder should look like the following:

```
data
  |--Frame_Labels
    |--Frame_Labels
      |--FACS
        |--042-ll042
        |-- ......
      |--PSPI
        |--042-ll042
        |-- ......
  |--Images
    |--Images
      |--042-ll042
      |-- ......
```

### Data Cleaning 

It cannot detect faces from the following images:

- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff002.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff005.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff006.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff007.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff008.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff009.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff019.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff022.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff025.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff028.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff039.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff052.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff070.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff241.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff242.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff243.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2afunaff\fn059t2afunaff244.png`
- Skip image: `data\Images\Images\059-fn059\fn059t2aiaff\fn059t2aiaff339.png`

The following images are pure black:

  - Skip image: `data\Images\Images\095-tv095\tv095t2aeunaff\tv095t2aeunaff001.png`
  - Skip image: `data\Images\Images\095-tv095\tv095t2aeunaff\tv095t2aeunaff002.png`
  - Skip image: `data\Images\Images\095-tv095\tv095t2aeunaff\tv095t2aeunaff003.png`
  - Skip image: `data\Images\Images\095-tv095\tv095t2aeunaff\tv095t2aeunaff004.png`
  - Skip image: `data\Images\Images\095-tv095\tv095t2aeunaff\tv095t2aeunaff005.png`
  - Skip image: `data\Images\Images\095-tv095\tv095t2aeunaff\tv095t2aeunaff006.png`
  - Skip image: `data\Images\Images\095-tv095\tv095t2aeunaff\tv095t2aeunaff007.png`

## Development Environment

| Operating System | CPU              | GPU                                  | CUDA Version | PyTorch Version |
| ---------------- | ---------------- | ------------------------------------ | ------------ | --------------- |
| Windows 10 Pro   | AMD Ryzen 7 1700 | NVIDIA GeForce GTX 1080 (8GB memory) | 11.6.124     | 1.10.0          |

### Install Packages

Run `pip install -r requirements.txt`

This will install all the necessary packages.

## How to run the code

### Step 1: Pre-process the data

Run `python process_data.py`

This will create a `data_summary.csv` file, it will fill the data into the following table:

| person_name | video_name   | frame_number | pspi_score | image_path                                                    |
| ----------- | ------------ | ------------ | ---------- | ------------------------------------------------------------- |
| 042-ll042   | ll042t1aaaff | 001          | 0.0        | data\Images\Images\042-ll042\ll042t1aaaff\ll042t1aaaff001.png |
| ...         | ...          | ...          | ...        | ...                                                           |

### Step 2: Run the main paper

Run `python train_and_test.py`

The `train_and_test.py` creates a `data.csv` from the `data_summary.csv` of step 1. It pairs each target image (PSPI>0) with a reference image (PSPI=0) of the same person. You can tune the number of randomly sampled reference images by changing the `image_sample_size`. Then, it splits the `data.csv` into training set (80%) and test set (20%). Finally, it trains the model on the training set and evaluate the model on the test set by calculating the average MSE loss, Pearson Correlation score, and the F1 score with the threshold equals to 2 (pain: PSPI>2, not pain: PSPI<=2>).

#### Hyper-parameters

You can tune the following hyper-parameters:

```
args_dict = {
    'image_scale_to_before_crop': 320,
    'image_size': 160,
    'number_output': 1,
    'image_sample_size': 5,
    'batch_size': 50,  
    'drop_out': 0,
    'fc2_size': 100,
    'learning_rate': 0.0001,
    'epoch': 5,
    # For testing the code
    'train_sample': 150,
    'test_sample': 50
}
```

### Step 3: Run the ExtendedMTL4Pain paper

Run `python train_and_test.py`

The `train_and_test.py` splits the `data_summary.csv` of step 1 into training set (80%) and test set (20%). Then, it trains the model on the training set and evaluate the model on the test set by calculating the average MSE loss, Pearson Correlation score, and the F1 score with the threshold equals to 2 (pain: PSPI>2, not pain: PSPI<=2>).

#### Hyper-parameters

You can tune the following hyper-parameters:

```
args_dict = {
    'image_scale_to_before_crop': 200,
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
```

### Train and test on part of the training data and test data

You can enable the following lines to sample a portion of the training and test data:

```
sample_data(TRAIN_DATA_CSV_PATH, ARGS.train_sample, RANDOM_SEED)
sample_data(TEST_DATA_CSV_PATH, ARGS.test_sample, RANDOM_SEED)
```

## References

Pytorch official document

- [https://pytorch.org/tutorials/beginner/data_loading_tutorial.html](https://pytorch.org/docs/stable/index.html)

Pandas official document

- [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html](https://pandas.pydata.org/docs/)

Course material 

- [https://uoft-csc413.github.io/2022/](https://uoft-csc413.github.io/2022/)

Programming assignment 4

- [https://colab.research.google.com/github/uoft-csc413/2022/blob/master/assets/assignments/a4_dcgan.ipynb](https://colab.research.google.com/github/uoft-csc413/2022/blob/master/assets/assignments/a4_dcgan.ipynb)

PyTorch YouTube tutorial 

- [https://youtu.be/BzcBsTou0C0](https://youtu.be/BzcBsTou0C0)
