import pandas as pd
from tensorflow import keras
from keras._tf_keras.keras.layers import Dense, Dropout, Flatten
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D



train_dir = "sign_mnist_train.csv"
test_dir = "sign_mnist_test.csv"
train_data = pd.read_csv(train_dir)
test_data = pd.read_csv(test_dir)

#Separating out the data and the labels
x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
x_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

#Preprocessing the input data
image_size = 28
x_train = x_train.reshape(-1, image_size, image_size, 1)
x_test = x_test.reshape(-1, image_size, image_size, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Converting the labels to one hot encoding
#This is hard coded due to the number of letters in asl being set and that 2 letters are missing
num_classes = 26
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


my_model = keras.Sequential([
        keras.Input(shape= [image_size, image_size, 1]),
        Conv2D(20, kernel_size=(5, 5), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(40, kernel_size=(5, 5), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(100),
        Dense(100),
        Dense(num_classes, activation="softmax"),
])

batch_size = 64
epochs = 25

my_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

my_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15)

score = my_model.evaluate(x_test, y_test, verbose=0)

print("Test loss:", score[0])
print("Test accuracy:", score[1])
my_model.save("my_model.keras")