# Federated Learning Deduplication

This repository contains an implementation of federated learning (FL) with non-IID (non-identically distributed) data. It utilizes a convolutional neural network (CNN) model for image classification tasks, particularly focusing on the MNIST dataset.

## Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)

## Introduction

The project considers the scenario of local client data duplication for federated learning and achieves deduplication of local data for federated learning under non-independent distribution of data, which shortens the training time of local models.

## Dependencies

This project requires the following dependencies:

- Python 3.12.3
- PyTorch
- matplotlib
- psutil

You can install the dependencies using pip:
```bash
pip install torch matplotlib psutil
```

## Usage
To use this codebase, follow these steps:

1.Clone this repository to your local machine:
```bash
git clone https://github.com/TangerineTree/fed-dedup.git
```
2.Navigate to the cloned directory:
```bash
cd fed-dedup
```
3.Run the main script to start federated learning:
```bash
python3 fed_avg.py
```
Or you can open the folder in Pycharm and simply run fed_avg.py, make sure that you have already installed the dependencies.

## Results
You can find the results of federated learning experiments in this section. Results include training time, CPU usage, memory usage, test accuracy, and loss curves.