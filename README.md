# End-to-End-Learning-for-Self-Driving-Cars
## Introduction

This project is a tensorflow implementation of [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). It trains an convolutional neural network (CNN) to learn a map from raw images to sterring command.

## Requirements

- Tensorflow >= r0.14
- opencv, numpy

## Howto

- Download the [dataset](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing)
- Train the model: `python train.py`
- Customize your params: `python train.py -h`

```shell
% python train.py --help
usage: train.py [-h] [--max_steps MAX_STEPS] [--print_steps PRINT_STEPS]
                [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                [--data_dir DATA_DIR] [--log_dir LOG_DIR]
                [--model_dir MODEL_DIR] [--seed SEED]
                [--train_prop TRAIN_PROP] [--validation_prop VALIDATION_PROP]
                [--disable_restore DISABLE_RESTORE]

optional arguments:
  -h, --help            show this help message and exit
  --max_steps MAX_STEPS
                        Number of steps to run trainer
  --print_steps PRINT_STEPS
                        Number of steps to print training loss
  --learning_rate LEARNING_RATE
                        Initial learning rate
  --batch_size BATCH_SIZE
                        Train batch size
  --data_dir DATA_DIR   Directory of data
  --log_dir LOG_DIR     Directory of log
  --model_dir MODEL_DIR
                        Directory of saved model
  --seed SEED           random seed to generate train, validation and test set
  --train_prop TRAIN_PROP
                        The proportion of train set in all data
  --validation_prop VALIDATION_PROP
                        The proportion of validation set in all data
  --disable_restore DISABLE_RESTORE
                        Whether disable restore model from model directory
```

- Visualize your training procedure: `tensorboard --logdir=./logs`

## Training Results

The model structure visualized by tensorboard:

![](./images/model.png)

The training loss:

![](./images/loss.png)

The test results:

```shell
Step 19200 train_loss:  0.0011946 validation_loss:  0.020691916285
Step 19300 train_loss:  0.000958753 validation_loss:  0.0227594176463
Step 19400 train_loss:  0.000808717 validation_loss:  0.0229436717168
Step 19500 train_loss:  0.000825631 validation_loss:  0.0214564389168
Step 19600 train_loss:  0.000872765 validation_loss:  0.0209440819074
Step 19700 train_loss:  0.000964281 validation_loss:  0.0220138165066
Step 19800 train_loss:  0.000996914 validation_loss:  0.0217930443876
Step 19900 train_loss:  0.000687053 validation_loss:  0.0206206155436
MAE in test dataset:  0.0600170250318
LOSS (MSE) in test dataset 0.0179578544099
```

# Acknoledge

Thanks to [Sully Chen](https://github.com/SullyChen) for the dataset.