# Train the model
import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history):
    accuracy = history.history['accuracy']
    loss = history.history['loss']

    epochs = range(1, len(accuracy) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, 'b', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'g', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


def trainer(model,x,y):
    history = model.fit(
        x,
        y,
        epochs=20,
        batch_size=16,
        verbose=1
    )

    model.save('/models/lstm_model.h5')

    # Plot and save training history
    plot_training_history(history)
