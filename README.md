## 1. Project Overview

Briefly introduce the topic of the paper, its objectives, and the core functionalities implemented. For example:

This project implements the core algorithms/models/experiments of the paper *XXXXX*, aiming to enhance the performance of YYY using the XXX approach. The code implements the ZZZ model and has been tested on various datasets.

## 2. Installation

### Required Environment and Dependencies

This project uses an `environment.yml` file to specify all dependencies for the project. You can set up the environment using Conda.

1. First, create the environment:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:

   ```bash
   conda activate your-environment-name
   ```

## 3. Dataset

### Dataset Name: 鸟鸣标记数据集（公开部分）

- Dataset Source: 百度网盘
- Download Link: [百度网盘链接](https://pan.baidu.com/s/1Z8sUKCSP6dy-OJapo7tm4Q?pwd=kths) 提取码: `kths`

### Dataset Preparation

1. Download the dataset from the provided link.

2. Extract the dataset to a local directory.

3. **Data Split**: According to the paper, the dataset should be split into three subsets: training set, validation set, and test set. The recommended split ratio is **6:2:2**, meaning:

   - **60%** of the data for training
   - **20%** of the data for validation
   - **20%** of the data for testing

   You can use a script to split the data or manually organize it into `train/`, `valid/`, and `test/` directories.

   Place the split datasets into the `data/` directory:

If any preprocessing is required, please provide relevant scripts or instructions.

## 4. Usage

We provide training scripts for four different models. You can train and test each model by running the corresponding script:

#### 1. **SGCN Model**

To train the SGCN model, run the following command:

```
python SGCN/main.py
```

#### 2. **baseGCN Model**

To train the baseGCN model, run the following command:

```
python baseGCN/main.py
```

#### 3. **EfficientNet Model**

To train the EfficientNet model, run the following command:

```
python EfficientNet.py
```

#### 4. **Hubert Model**

To train the Hubert model, run the following command:

```
python Hubert.py
```

### Configuration File

You can modify the `config.py` file to adjust hyperparameters, paths, and other configurations.

## 5. License

This project is licensed under the [MIT License](https://chatgpt.com/c/LICENSE).

------

