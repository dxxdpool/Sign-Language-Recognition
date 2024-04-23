import keras.models
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Constants
num_classes = 3 # Number of classes in your dataset
image_size = (300, 300)  # Size of input images
batch_size = 15  # Batch size for training
epochs = 25 # Number of epochs

# Data preparation
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory='ImageData/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=42,
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory='ImageData/test',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False,
    seed=42
)

valid_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory='ImageData/valid',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False,
    seed=42
)


def TrainModel(train_data):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=image_size + (3,))
    for layer in base_model.layers:
        layer.trainable = False

    # Add global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a new dense layer with number of classes as output units
    x = Dense(len(train_data.class_names), activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model for 10 epochs
    H = model.fit(train_data, epochs=epochs, validation_data=valid_data)

    model.save('SignModel.keras')
    history_df = pd.DataFrame(H.history)

    # Save the DataFrame to an Excel file
    history_df.to_excel('training_history.xlsx', index=False)


def TestModel(test_data):
    model = keras.models.load_model('SignModel.keras')
    y_pred = np.argmax(model.predict(test_data), axis=-1)
    y_true = np.concatenate([y for x, y in test_data], axis=0)
    y_true = np.argmax(y_true, axis=1)

    print(y_pred)
    print(y_true)

    # Generate classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Generate confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    history_df = pd.read_excel('training_history.xlsx')

    # Extract loss and validation loss
    loss = history_df['loss']
    val_loss = history_df['val_loss']

    # Plot loss and validation loss versus epochs
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss', color = 'red')
    plt.plot(epochs, val_loss, 'b', label='Validation loss', color='blue')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# TrainModel(train_data)
TestModel(test_data)