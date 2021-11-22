from model.classifier.prepare_data import get_train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical

x_train, x_test, y_train, y_test = get_train_test_split()

# Scale the data
x_train, x_test = x_train.astype('float32') / 255., x_test.astype('float32') / 255.

# Convert label array to one hot encoded arrays
y_train = to_categorical(y_train, 5)
y_test = to_categorical(y_test, 5)

print("\nx_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("\nx_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))


model = models.Sequential()

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(100, 100, 3), name='conv_1'))
model.add(layers.MaxPool2D(pool_size=(2, 2), name='pool_1'))

model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_2'))
model.add(layers.MaxPool2D(pool_size=(2, 2), name='pool_2'))

model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_3'))
model.add(layers.MaxPool2D(pool_size=(2, 2), name='pool_3'))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu', name='dense_1'))
model.add(layers.Dropout(rate=.5))

model.add(layers.Dense(64, activation='relu', name='dense_2'))
model.add(layers.Dropout(rate=.5))

model.add(layers.Dense(units=5, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer= optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=32, verbose=1)

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

model.save('sign_digit_classifier.h5')
