# Multiclass LSTM Classifier for ADFA Windows Dataset (ADFA-WD)

## Overview
This project implements a Multiclass Classifier using Long-Short Term Memory (LSTM) model in Tensorflow to detect malicious system processes in the ADFA Windows Dataset. The model utilizes n-gram based analysis combined with LSTM/Bi-LSTM neural networks and Word2Vec embeddings for accurate attack detection.

## Dataset
The ADFA Windows Dataset (ADFA-WD) is a contemporary Windows-based dataset specifically designed for HIDS evaluation. It can be downloaded from:
[ADFA-IDS Datasets](https://research.unsw.edu.au/projects/adfa-ids-datasets)
[Alternative link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IFTZPF)

### Project Structure
```
-deployment/
├── data/
    ├── training_data /
    └── test_data/
├── results/
├── models/
├── data_loader.py/
├── data_preprocessing.py/
├── model.py/
├── evaluate.py/
├── train.py/
├── evaluate.py/
├── main.py/
└── requirements.txt/
-jupyter_notebooks/
-readme.md/
```

## Project Objectives
1. Development of an n-gram based multi-class classifier for intrusion detection.
2. Implementation of a Bi-LSTM architecture.
3. Utilization Word2Vec embeddings for n-gram representation.


## Experimental Analysis
1. The experiments were originally performed on a Google colab server with 350 GB RAM and T4 TPU with 32 GB TPU support.
2. The .ipynb notebooks for these can be find in the `jupyter_notebook` folder.
3. Also, one can run the experiments on servers by using the following instructions.



## Setup and Installation

### Prerequisites
```bash
python >= 3.8
tensorflow >= 2.0
gensim >= 4.0
numpy >= 1.19
pandas >= 1.2
scikit-learn >= 0.24
matplotlib
seaborn
```

### Installation
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and extract the ADFA-WD dataset to the `data/` directory inside the `deployment/`

## Usage for deployment on servers

1. To train and test the model run main.py
2. To make predictions please use predictor.py (You need to have a model saved in models folder for it to train)


## Model Configuration
- N-gram size: 3 (configurable)
- Embedding dimensions: 50/100
- LSTM/Bi-LSTM units: 128
- Dropout rate: 0.2
- Training epochs: 50
- Batch size: 32

## Performance Metrics
The model's performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score


## Citation
If you use this implementation, please cite:
```
[1] G. Creech. "Developing a high-accuracy cross platform Host-Based Intrusion
    Detection System capable of reliably detecting zero-day attacks", 2014
```


## Acknowledgments
- ADFA-WD dataset creators and maintainers
- UNSW-Canberra for hosting the dataset
- Harvard Dataverse for hosting the dataset as the alternative site
- Contributors to the original research paper
