import numpy as np
import tensorflow.keras as K
import pickle
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, classification_report)
import matplotlib.pyplot as plt

# Load your eczema test data and labels
x_test = pickle.load(open('D:\\preprocess\\x_test.pkl', 'rb'))
y_test = pickle.load(open('D:\\preprocess\\y_test.pkl', 'rb'))




# Load your model and training history
model = K.models.load_model("D:/allergydetection/second/model.h5")


# Load the training history from the .h5 file
with open('D:/allergydetection/second/hist.pkl', "rb") as file_pi:
    hist = pickle.load(file_pi)

# Define your class names
class_names = ['eczema', 'normal']

# Make predictions on the test data
predictions = model.predict(x_test)
test_pred = np.argmax(predictions, axis=1)

# Create a confusion matrix
cm = confusion_matrix(y_test, test_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.show()

# Plot accuracy vs. epoch
plt.plot(hist['accuracy'])
plt.plot(hist['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()
plt.show()

# Plot loss vs. epoch
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()
plt.show()

# Print classification report
print("\nClassification Report: \n" + str(classification_report(y_test, test_pred)))
