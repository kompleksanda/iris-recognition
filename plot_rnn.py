import pickle
import matplotlib.pyplot as plt

print("Loading model's history (before filter)...")
historyBF = pickle.load(open("Iris_SVD_RNN.pickle", "rb"))

print("Plotting graphs...")
# Plot training & validation accuracy values
fig1 = plt.figure(1)
plt.plot(historyBF['accuracy'])
plt.plot(historyBF['val_accuracy'])
plt.title('Model Accuracy Before Filter')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='center right')
plt.savefig('Iris_Accuracy.png')

# Plot training & validation loss values
plt.figure(2)
plt.plot(historyBF['loss'])
plt.plot(historyBF['val_loss'])
plt.title('Model Loss Before Filter')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='center right')
plt.savefig('iris_Loss.png')

