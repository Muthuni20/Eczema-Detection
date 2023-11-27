import pickle
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from keras.applications import Xception
from keras.models import Model
import tensorflow.keras as K

x_train = pickle.load(open('D:\\augment\\x_train', 'rb'))
y_train = pickle.load(open('D:\\augment\\y_train', 'rb'))
x_val = pickle.load(open('D:\\augment\\x_val', 'rb'))
y_val = pickle.load(open('D:\\augment\\y_val', 'rb'))
x_test = pickle.load(open('D:\\augment\\x_test', 'rb'))
y_test = pickle.load(open('D:\\augment\\y_test', 'rb'))

base_model = Xception(include_top=False,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=(299, 299, 3),
                      pooling=None,
                      classifier_activation="softmax")

base_model.trainable = False

inputs = K.Input(shape=(299, 299, 3))

x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

outputs = Dense(1, activation='sigmoid')(x)  # final layer for binary classification (eczema or not)

model = Model(inputs, outputs)

opt = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy',  # change loss function for binary classification
              optimizer=opt,
              metrics=['accuracy'])
acc_checkpoint = ModelCheckpoint("D:/allergydetection/max_acc", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
loss_checkpoint = ModelCheckpoint("D:/allergydetection/min_loss", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [acc_checkpoint, loss_checkpoint]

hist = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=32, callbacks=callbacks_list)


model.save("D:/allergydetection/second/model.h5")


with open('D:/allergydetection/second/hist.pkl', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
