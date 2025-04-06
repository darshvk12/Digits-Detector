import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Decide whether to train a new model or load an existing one
train_new_model = True

if train_new_model:
    # Loading the MNIST data set
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (pixel values between 0 and 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Creating the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compiling the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=3)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print("Validation loss:", val_loss)
    print("Validation accuracy:", val_acc)

    # Saving the model
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/handwritten_digits.model.h5')

else:
    # Load the existing model
    model = tf.keras.models.load_model('models/handwritten_digits.model.h5')

# Predict custom images stored in 'digits' directory
image_number = 1
while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        img = cv2.imread(f'digits/digit{image_number}.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))  # Ensure image is 28x28
        img = np.invert(img)  # Invert colors
        img = img / 255.0  # Normalize the image
        img = img.reshape(1, 28, 28)  # Reshape for prediction

        prediction = model.predict(img)
        print(f"The number is probably a {np.argmax(prediction)}")

        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(prediction)}")
        plt.show()
        image_number += 1

    except Exception as e:
        print(f"Error reading image {image_number}: {e}")
        image_number += 1
