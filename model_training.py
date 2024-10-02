import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops.gen_experimental_dataset_ops import LoadDataset

from config import IMAGE_SIZE, CHANNELS
from .data_preprocessing import load_dataset, get_dataset_partitions, preprocess_datasets
from .config import DATASET_PATH, EPOCHS, N_CLASSES, BATCH_SIZE

def create_model(input_shape : tuple[int,int,int]):
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(input_shape[1],input_shape[1]),
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(N_CLASSES, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model() -> tf.keras.Model:
    dataset = LoadDataset(DATASET_PATH)
    train_dataset,val_dataset, test_dataset = get_dataset_partitions(dataset)
    train_dataset,val_dataset, test_dataset = preprocess_datasets(train_dataset, val_dataset, test_dataset)

    input_shape=(IMAGE_SIZE, IMAGE_SIZE,CHANNELS)
    model = create_model(input_shape)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        verbose=1,
    )
    return model, history