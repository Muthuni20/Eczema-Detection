import pickle
import numpy as np
from tqdm import tqdm
from skimage.transform import rotate

# Load your custom datasets from the specified paths
x_train = pickle.load(open('D:\\preprocess\\x_train.pkl', 'rb'))
y_train = pickle.load(open('D:\\preprocess\\y_train.pkl', 'rb'))
x_val = pickle.load(open('D:\\preprocess\\x_val.pkl', 'rb'))
y_val = pickle.load(open('D:\\preprocess\\y_val.pkl', 'rb'))
x_test = pickle.load(open('D:\\preprocess\\x_test.pkl', 'rb'))
y_test = pickle.load(open('D:\\preprocess\\y_test.pkl', 'rb'))

x_t = len(x_train)
x_copy = x_train.copy()

# Initialize counters for different skin conditions
eczema_count = 0
normal_skin_count = 0

# Define the augmentation strategy for different conditions
for i in tqdm(range(x_t)):
    if y_train[i] == 0:  # Assuming class 0 corresponds to "normal_skin"
        # Perform augmentation for normal skin photos
        x_train = np.append(x_train, [rotate(x_copy[i], angle=45, mode='wrap')], axis=0)
        y_train = np.append(y_train, 0)
        normal_skin_count += 1

    elif y_train[i] == 1:  # Assuming class 1 corresponds to "eczema"
        if eczema_count % 3 == 0:
            # Perform augmentation for eczema photos
            x_train = np.append(x_train, [rotate(x_copy[i], angle=45, mode='wrap')], axis=0)
            x_train = np.append(x_train, [np.flipud(x_copy[i])], axis=0)
            for j in range(2):
                y_train = np.append(y_train, 1)  # Assuming 1 corresponds to "eczema"
            eczema_count += 1

# Shuffle the augmented data
indices = np.random.permutation(len(x_train))
x_train = x_train[indices]
y_train = y_train[indices]



num_samples_x_train = len(x_train)
num_samples_y_train = len(y_train)



# Save the augmented datasets
f = open('D:\\augment\\x_train', 'wb')
pickle.dump(x_train, f)
f.close()

f = open('D:\\augment\\y_train', 'wb')
pickle.dump(y_train, f)
f.close()

f = open('D:\\augment\\x_val', 'wb')
pickle.dump(x_val, f)
f.close()

f = open('D:\\augment\\y_val', 'wb')
pickle.dump(y_val, f)
f.close()

f = open('D:\\augment\\x_test', 'wb')
pickle.dump(x_test, f)
f.close()

f = open('D:\\augment\\y_test', 'wb')
pickle.dump(y_test, f)
f.close()
