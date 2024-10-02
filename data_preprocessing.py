import tensorflow as tf
from typing import Tuple


from config import IMAGE_SIZE,BATCH_SIZE

def load_dataset(path: str):
    """Load the dataset from the specified path."""
    try:
        return tf.keras.preprocessing.image_dataset_from_directory(path,
        shuffle=True,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE
        )
    except Exception as e:
        raise ValueError(f"Error while loading the dataset: {e}")


def get_dataset_partitions(dataset: tf.data.Dataset,
                           train_split: float = 0.8,
                           val_split: float = 0.1,
                           test_split: float = 0.1,
                           shuffle: bool = True,
                           shuffle_size: int = 10000) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Partition the dataset into training, validation, and test sets.

    Args:
        dataset (tf.data.Dataset): The dataset to partition.
        train_split (float): Proportion of data for training.
        val_split (float): Proportion of data for validation.
        test_split (float): Proportion of data for testing.
        shuffle (bool): Whether to shuffle the dataset.
        shuffle_size (int): Size of the shuffle buffer.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Train, validation, and test datasets.
    """
    dataset_size = len(dataset)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    return train_dataset, val_dataset, test_dataset

def preprocess_datasets(train_dataset: tf.data.Dataset,
                        val_dataset: tf.data.Dataset,
                        test_dataset: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Preprocess the datasets by caching, shuffling, and prefetching.

    Args:
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        test_dataset (tf.data.Dataset): The test dataset.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Preprocessed train, validation, and test datasets.
    """
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
