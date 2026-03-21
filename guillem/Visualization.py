import matplotlib.pyplot as plt
import os


def Visualization_plots(history):
    os.makedirs("plots", exist_ok=True)

    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy (Last Fold)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/accuracy.png")
    plt.close()
    print("Saved: plots/accuracy.png")

    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss (Last Fold)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig("plots/loss.png")
    plt.close()
    print("Saved: plots/loss.png")