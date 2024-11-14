import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def tester(model, test_data, test_labels, num_classes=13):
    loss, test_accuracy = model.evaluate(test_data, test_labels)
    print("Test accuracy :", test_accuracy)
