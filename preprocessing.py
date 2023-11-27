


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

dataset_root = r'D:\Research'
# Define classes and corresponding labels
class_names = ['eczema', 'normal_skin']
class_labels = [1, 0]  # Eczema photos are labeled as 1, normal skin photos as 0

# Initialize lists to store features (images) and labels
features = []
labels = []

# Load both eczema and normal_skin photos
for class_label, class_name in zip(class_labels, class_names):
    class_dir = os.path.join(dataset_root, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(299, 299))
        x = img_to_array(img)
        x = preprocess_input(x)

        # Append the preprocessed image and its label to the lists
        features.append(x)
        labels.append(class_label)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Shuffle the data
indices = np.random.permutation(len(features))
features = features[indices]
labels = labels[indices]

# Split the data into training, validation, and test sets
x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

# Split the test set
split_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in split_test.split(features, labels):
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# Split the training data into training and validation sets
split_train = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
for train_index, val_index in split_train.split(x_train, y_train):
    x_train, x_val = x_train[train_index], x_train[val_index]
    y_train, y_val = y_train[train_index], y_train[val_index]

# Visualize the dataset distribution
data = [len(x_train), len(x_val), len(x_test)]
labels = ['train', 'val', 'test']

colors = sns.color_palette('pastel')[0:5]

plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')  # Create a pie chart
plt.show()

# Visualize the class distribution in train, validation, and test sets
lbl_train, count_train = np.unique(y_train, return_counts=True)
lbl_val, count_val = np.unique(y_val, return_counts=True)
lbl_test, count_test = np.unique(y_test, return_counts=True)

temp = {1: 'eczema', 0: 'normal_skin'}


train_dict, val_dict, test_dict = {}, {}, {}
for i in range(len(lbl_train)):
    train_dict[temp[lbl_train[i]]] = count_train[i]
for i in range(len(lbl_val)):
    val_dict[temp[lbl_val[i]]] = count_val[i]
for i in range(len(lbl_test)):
    test_dict[temp[lbl_test[i]]] = count_test[i]

lst_lbl, lst_count, lst_type = [], [], []
for k, v in train_dict.items():
    lst_lbl.append(k)
    lst_type.append('train')
    lst_count.append(v)
for k, v in val_dict.items():
    lst_lbl.append(k)
    lst_type.append('validation')
    lst_count.append(v)
for k, v in test_dict.items():
    lst_lbl.append(k)
    lst_type.append('test')
    lst_count.append(v)

sns.set_theme(style="whitegrid")
ax = sns.barplot(x=lst_type, y=lst_count, hue=lst_lbl)
for i in ax.containers:
    ax.bar_label(i, )
plt.show()

# Save the datasets as pickle files
data_dir = "D:\preprocess"  # Replace with the directory where you want to save the datasets

with open(os.path.join(data_dir, 'x_train.pkl'), 'wb') as file:
    pickle.dump(x_train, file)
with open(os.path.join(data_dir, 'y_train.pkl'), 'wb') as file:
    pickle.dump(y_train, file)

with open(os.path.join(data_dir, 'x_val.pkl'), 'wb') as file:
    pickle.dump(x_val, file)
with open(os.path.join(data_dir, 'y_val.pkl'), 'wb') as file:
    pickle.dump(y_val, file)

with open(os.path.join(data_dir, 'x_test.pkl'), 'wb') as file:
    pickle.dump(x_test, file)
with open(os.path.join(data_dir, 'y_test.pkl'), 'wb') as file:
    pickle.dump(y_test, file)



import cv2
import numpy as np

def calculate_image_quality(image1, image2):
    # Load images
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # Ensure images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    # Calculate SSIM
    ssim = cv2.SSIM(img1, img2)

    # Calculate PSNR
    mse = np.mean((img1 - img2) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse)

    return ssim, psnr

if __name__ == "__main__":
    image1_path = "path_to_image1.jpg"  
    image2_path = "path_to_image2.jpg"  

    ssim, psnr = calculate_image_quality(image1_path, image2_path)

    print(f"SSIM: {ssim}")
    print(f"PSNR: {psnr}")

