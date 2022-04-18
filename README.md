# Deep Learning Project: Song Recommendation System

## 1. Getting Started
### 1.1 Dataset preparation
1. Download the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) 
2. Remove `jazz.00055...` row from the `features_30_sec.csv` file.
3. Directory structure should be as such:

    ```
    .
    ├── checkpoints
    │   └── cnn
    ├── data
    │   └── images_original
    │       ├── blues
    │       ├── classical
    │       ├── country
    │       ├── disco
    │       ├── hiphop
    │       ├── jazz
    │       ├── metal
    │       ├── pop
    │       ├── reggae
    │       └── rock
    ├── img
    ├── notebooks
    │   ├── deep_learning_base_code.ipynb
    │   ├── deep_learning_base_code_inception_module\ (1).ipynb
    │   ├── deep_learning_base_code_with_song_embedding.ipynb
    │   ├── deep_learning_cnn.ipynb
    │   └── hats_deep_learning_cnn.ipynb
    ├── requirements.txt
    └── src
        ├── config
        │   ├── __pycache__
        │   │   ├── eval_config.cpython-39.pyc
        │   │   └── train_config.cpython-39.pyc
        │   ├── eval_config.py
        │   └── train_config.py
        ├── evaluate.py
        ├── models
        │   ├── CustomCNN.py
        │   ├── InceptionModule.py
        │   ├── ResNet.py
        │   └── __pycache__
        │       ├── CustomCNN.cpython-39.pyc
        │       ├── InceptionModule.cpython-39.pyc
        │       └── ResNet.cpython-39.pyc
        ├── train.py
        └── utils
            ├── __pycache__
            │   ├── eval_utils.cpython-39.pyc
            │   ├── model_utils.cpython-39.pyc
            │   ├── preprocessing_utils.cpython-39.pyc
            │   └── train_utils.cpython-39.pyc
            ├── eval_utils.py
            ├── model_utils.py
            ├── preprocessing_utils.py
            └── train_utils.py
    ```

### 1.2 Setup environment    
Install all the required dependencies
    ```
    pip3 install -r requirements.txt
    ```

### 1.3 Change working directory path
1. Open `config/train_configs` and `config/eval_configs`
2. Change the default `--working_dir` in `line 9` to your own **absolute path**  or specify it during training/evaluation

## 2. How to run
### 2.1 Training
1. Change directory to `/src`
    ```
    cd src
    ```

2. Run training code
    ```
    python3 train.py --no_cuda
    ```

    ```
    $ python3 train.py --help
    usage: train.py [-h] [--working_dir WORKING_DIR] [-a ARCH] [--checkpoints_path CHECKPOINTS_PATH] [--test_size TEST_SIZE] [--num_workers NUM_WORKERS]
                [--trg_batch_size TRG_BATCH_SIZE] [--val_batch_size VAL_BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                [--lr LR] [--momentum MOMENTUM] [--gamma GAMMA] [--optimizer_type OPTIMIZER_TYPE] [--step_size STEP_SIZE] [--no_cuda]

    optional arguments:
    -h, --help            show this help message and exit
    --working_dir WORKING_DIR
                            The ROOT working directory
    -a ARCH, --arch ARCH  The architecture of model
    --checkpoints_path CHECKPOINTS_PATH
                            The path of the pretrained checkpoint
    --test_size TEST_SIZE
                            Train test split
    --num_workers NUM_WORKERS
                            Number of threads for loading data
    --trg_batch_size TRG_BATCH_SIZE
                            Batch size
    --val_batch_size VAL_BATCH_SIZE
                            Batch size
    --test_batch_size TEST_BATCH_SIZE
                            Batch size
    --num_epochs NUM_EPOCHS
                            Number of epochs to train
    --lr LR               Learning rate
    --momentum MOMENTUM   Momentum
    --gamma GAMMA         Gamma
    --optimizer_type OPTIMIZER_TYPE
                            Type of optimizer: sgd or adam
    --step_size STEP_SIZE
                            Step size
    --no_cuda             If true, cuda is not used.
    ```

### 2.2 Evaluation
1. Change directory to `/src`
    ```
    cd src
    ```

2. Run evaluation code
    ```
    python3 evaluate.py --no_cuda
    ```
    
    ```
    $ python3 evaluate.py --help
    usage: evaluate.py [-h] [--working_dir WORKING_DIR] [-a ARCH] [--test_size TEST_SIZE] [--num_workers NUM_WORKERS] [--trg_batch_size TRG_BATCH_SIZE]
                   [--val_batch_size VAL_BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE] [--no_cuda]

    optional arguments:
    -h, --help            show this help message and exit
    --working_dir WORKING_DIR
                            The ROOT working directory
    -a ARCH, --arch ARCH  The architecture of model
    --test_size TEST_SIZE
                            Train test split
    --num_workers NUM_WORKERS
                            Number of threads for loading data
    --trg_batch_size TRG_BATCH_SIZE
                            Batch size
    --val_batch_size VAL_BATCH_SIZE
                            Batch size
    --test_batch_size TEST_BATCH_SIZE
                            Batch size
    --no_cuda             If true, cuda is not used.
    ```
