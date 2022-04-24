# Deep Learning Project: Song Recommendation System

## 1. Getting Started
### 1.1 Dataset preparation
1. Download the [GTZAN Dataset preprocessed](https://drive.google.com/file/d/1rUuelL3TxKNrF_5j9gV6Z4fgb0cIxcWD/view?usp=sharing) 
2. Visualize an individual audio file

-Spectral Centroid:
![Test Image 1](https://github.com/mihirchhiber/deep_learning_project/blob/main/img/spectral%20centroid%20of%20blue1.png)

-Sound Wave:
![Test Image 2](https://github.com/mihirchhiber/deep_learning_project/blob/main/img/sound%20wave%20of%20blue1.png)
3. Directory structure should be as such:

    ```
    .
    ├── checkpoints
    │   ├── cnn
    │   ├── gru
    │   ├── incp
    │   ├── lstm
    │   ├── resnet
    │   └── rnn
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
    │   ├── Audio_Preprocessing.ipynb
    │   ├── Basecode.ipynb
    │   ├── CustomCNN_HyperparameterTuning.ipynb
    │   ├── CustomCNN.ipynb
    │   ├── Dataset_Visualisation.ipynb
    │   ├── deep_learning_song_embedding.ipynb
    │   ├── InceptionModule.ipynb
    │   ├── Initial_Recommender_System.ipynb
    │   ├── RecurrentNet.ipynb
    │   ├── ResNet.ipynb
    │   └── Song_Embeddings.ipynb
    ├── requirements.txt
    └── src
        ├── config
        │   ├── __pycache__
        │   │   ├── eval_config.cpython-39.pyc
        │   │   ├── preprocess_dataset_config.cpython-39.pyc
        │   │   ├── song_embed_config.cpython-39.pyc
        │   │   └── train_config.cpython-39.pyc
        │   ├── eval_config.py
        │   └── train_config.py
        ├── evaluate_song_embed.py
        ├── evaluate.py
        ├── extract_song_embed.py
        ├── inference.py
        ├── models
        │   ├── CustomCNN.py
        │   ├── InceptionModule.py
        │   ├── ResNet.py
        │   └── __pycache__
        │       ├── CustomCNN.cpython-39.pyc
        │       ├── InceptionModule.cpython-39.pyc
        │       └── ResNet.cpython-39.pyc
        ├── train.py
        ├── preprocess_dataset.py
        └── utils
            ├── __pycache__
            │   ├── eval_utils.cpython-39.pyc
            │   ├── extract_song_embed.cpython-39.pyc
            │   ├── inference_utils.cpython-39.pyc
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

### 1.4 Model Weights Guide
    <TO DO>

## 2. How to run
### 2.1 Training model
1. Change directory to `/src`
    ```
    cd src
    ```

2. Run training code
    ```
    python3 train.py --no_cuda --working_dir <ABS PATH>/deep_learning_project --arch cnn
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

    For CNN approach:
    ```
    python3 train.py --no_cuda --working_dir <ABS PATH>/deep_learning_project --arch cnn
    ```
    For InceptionModule approach:
    ```
    python3 train.py --no_cuda --working_dir <ABS PATH>/deep_learning_project --arch incp
    ```
    For ResNet approach:
    ```
    python3 train.py --no_cuda --working_dir <ABS PATH>/deep_learning_project --arch resnet
    ```
    For RecurrentNet approach:

    (RNN)
    ```
    python3 train.py --no_cuda --working_dir <ABS PATH>/deep_learning_project --arch rnn
    ```
    
    (LSTM)
    ```
    python3 train.py --no_cuda --working_dir <ABS PATH>/deep_learning_project --arch lstm
    ```
    
    (GRU)
    ```
    python3 train.py --no_cuda --working_dir <ABS PATH>/deep_learning_project --arch gru
    ```


### 2.2 Evaluation of model
1. Change directory to `/src`
    ```
    cd src
    ```

2. Run evaluation code
    ```
    python3 evaluate.py --no_cuda --working_dir <ABS PATH>/deep_learning_project --arch cnn
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
### 2.3 Extracting song embedding
1. Change directory to `/src`
    ```
    cd src
    ```

2. Run evaluation code
    ```
    python3 extract_song_embed.py --no_cuda --working_dir <ABS PATH>/deep_learning_project
    ```
    
    $ python3 extract_song_embed.py --help
    usage: train.py [-h] [--working_dir WORKING_DIR] [--test_size TEST_SIZE] [--num_workers NUM_WORKERS]
                [--trg_batch_size TRG_BATCH_SIZE] [--val_batch_size VAL_BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                [--lr LR] [--momentum MOMENTUM] [--gamma GAMMA] [--optimizer_type OPTIMIZER_TYPE] [--step_size STEP_SIZE] [--no_cuda]

    optional arguments:
    -h, --help            show this help message and exit
    --working_dir WORKING_DIR
                            The ROOT working directory
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

### 2.4 Evaluation of model
1. Change directory to `/src`
    ```
    cd src
    ```

2. Run evaluation code
    ```
    python3 evaluate_song_embedding.py --no_cuda --working_dir <ABS PATH>/deep_learning_project
    ```
    
### 2.3 Inference
1. Change directory to `/src`
    ```
    cd src
    ```
2. Run inference code
    ```
    python3 inference.py --no_cuda --working_dir <ABS PATH>deep_learning_project
    ```
