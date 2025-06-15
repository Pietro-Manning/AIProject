import os

import tensorflow as tf
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from config.config_loader import CONFIG
from src.utils.dataframe_utils import replace_characters
from utils import utils

def get_and_preprocess_mnds(dim:int = None):
    """
    Fetches and preprocesses a multi-label dataset for use in machine learning applications.
    The function determines if a pre-processed dataset is available. If it exists, the
    dataset is directly loaded from the disk. Otherwise, raw data is downloaded and processed to
    generate a structured dataset, which is then saved for future use. Additionally,
    category labels are handled and converted into a format suitable for training deep learning
    models.

    :param dim: Specifies the number of samples to include in the "train" dataset split. If None,
        the entire dataset is used.
    :type dim: int, optional
    :return: A tuple consisting of a TensorFlow dataset and a list of category class names. The
        TensorFlow dataset contains samples with input text and corresponding one-hot encoded
        categories.
    :rtype: tuple[tf.data.Dataset, list[str]]
    """
    dataset_raw_dir = CONFIG["datasets_processed_path"]
    dataset_path = os.path.join(dataset_raw_dir, f"mnds{'_' + str(dim) if dim is not None else ''}.json")

    if os.path.exists(dataset_path):
        df_final = pd.read_json(dataset_path, orient="records", lines=True)
        df_final_text = df_final["text"].astype(str).values
        df_final["category"] = df_final["category"].apply(lambda x: np.array(x, dtype=np.int32))
        df_final_categories = np.array(df_final["category"].tolist(), dtype=np.int32)

        dataset = tf.data.Dataset.from_tensor_slices((df_final_text, df_final_categories))
        classes_names = utils.get_class_names("C:\\Users\pietr\AIProject\data\processed\\")
        return dataset, classes_names

    else:

        if dim is not None: split = f"train[:{dim}]"
        else: split = "train"

        mlb = MultiLabelBinarizer()

        dataset_raw_dir = CONFIG["datasets_raw_path"]
        dataset_row_path = os.path.join(dataset_raw_dir, f"mnds{'_' + str(dim) if dim is not None else ''}.csv")

        if os.path.exists(dataset_row_path):
            df = pd.read_csv(dataset_row_path)
        else:
            ds = load_dataset("textminr/mn-ds", split=split)
            ds.to_csv(dataset_row_path, index=False)
            df = pd.DataFrame(ds)

        df.rename(columns={"content": "text"}, inplace=True)
        df = df[['text', 'category_level_1', 'category_level_2']]
        df['category'] = None

        df_final = replace_characters(df)

        category = (df_final[['category', 'category_level_1', 'category_level_2']].fillna("").values.tolist())
        category = [[el for el in _list if el != ""] for _list in category]

        category_encoded = mlb.fit_transform(category)

        df_final['category'] = list(category_encoded)
        df_final = df_final[['text', 'category']].copy()

        df_final_text = df_final["text"].astype(str).values
        df_final_categories = np.array(df_final["category"].tolist(), dtype=np.int32)

        dataset = tf.data.Dataset.from_tensor_slices((df_final_text, df_final_categories))
        utils.write_classes_to_file(mlb.classes_, "C:\\Users\pietr\AIProject\data\processed\\")
        classes_names = mlb.classes_

        df_final.to_json(dataset_path, orient='records', lines=True)

        return dataset, classes_names

def split_dataset(dataset, test_size:float = 0.1, val_size:float = 0.1, random_state:int = 42):
    """
    Splits a dataset into training, validation, and testing subsets. The function is
    compatible with both Pandas DataFrame and TensorFlow dataset objects. It ensures
    the provided dataset is divided according to the specified proportions for
    testing and validation, while maintaining the remaining portion for training.
    The function raises appropriate type and value exceptions for incorrect inputs.

    :param dataset: The input dataset to be split. Can either be a Pandas DataFrame
       or a TensorFlow Dataset.
       type: pd.DataFrame | tf.data.Dataset
    :param test_size: Proportion of the dataset to allocate for testing. Must be a
       float between 0.0 and 1.0.
       type: float
    :param val_size: Proportion of the dataset to allocate for validation. Must be
       a float between 0.0 and 1.0.
       type: float
    :param random_state: Seed for random number generator, used to ensure
       reproducible splits. Must be an integer.
       type: int
    :return: A tuple containing three datasets: training, validation, and testing
       subsets. The specific structure and type of the return depend on the type
       of the input dataset.
       type: Tuple[pd.DataFrame | tf.data.Dataset, pd.DataFrame | tf.data.Dataset,
       pd.DataFrame | tf.data.Dataset]
    """
    if not isinstance(dataset, pd.DataFrame) and not isinstance(dataset, tf.data.Dataset): raise TypeError("dataset must be either a Pandas DataFrame or a TensorFlow dataset")
    if not isinstance(test_size, float) or not isinstance(val_size, float): raise TypeError("test_size and val_size must be floats")
    if not isinstance(random_state, int): raise TypeError("random_state must be an integer")
    if test_size < 0.0 or test_size > 1.0: raise ValueError("test_size must be between 0.0 and 1.0")
    if val_size < 0.0 or val_size > 1.0: raise ValueError("val_size must be between 0.0 and 1.0")
    if random_state < 0: raise ValueError("random_state must be a positive integer")

    # splitting the data frame
    if isinstance(dataset, pd.DataFrame): return split_df_dataset(dataset, random_state, test_size, val_size)

    # splitting the tf.data.Dataset:
    if isinstance(dataset, tf.data.Dataset): return split_tf_dataset(dataset, test_size, val_size, random_state)

    return None, None, None

def split_df_dataset(dataset, random_state, test_size, val_size):
    """
    Splits a dataset into three parts: training, validation, and testing datasets. This function first splits
    the dataset into testing and the remaining data. Then, it further splits the remaining data into training
    and validation datasets based on the provided sizes.

    :param dataset:
        A pandas DataFrame or similar dataset object to be split.
    :param random_state:
        A seed value for the random number generator to ensure replicability of the split.
    :param test_size:
        A float value representing the proportion of the dataset to be allocated to the test set.
        This value should lie between 0 and 1.
    :param val_size:
        A float value representing the proportion of the dataset to be allocated to the validation
        set, relative to the original dataset. This value should lie between 0 and 1.
    :return:
        A tuple containing three datasets:
        - df_train: The training dataset.
        - df_val: The validation dataset.
        - df_test: The testing dataset.
    """
    # First, we split 'test' from the rest
    df_train_val, df_test = train_test_split(
        dataset, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Second, we calculate 'val_size' relative 'to df_train_val'
    # (1-'test_size'):100='val_size':x
    # (100 * 'val_size') / (1-'test_size')
    val_relative = val_size / (1.0 - test_size)

    df_train, df_val = train_test_split(
        df_train_val, test_size=val_relative, random_state=random_state, shuffle=True
    )
    return df_train, df_val, df_test

def split_tf_dataset(dataset:tf.data.Dataset, test_size=0.1, val_size=0.1, random_seed=42, buffer_size=10000):
    """
    Splits a TensorFlow dataset into train, validation, and test sets based on the specified
    ratios. The dataset is shuffled before splitting to ensure randomness.

    :param dataset: A tf.data.Dataset object to be split.
    :param test_size: A float representing the proportion of the dataset to be allocated
        to the test set.
    :param val_size: A float representing the proportion of the dataset to be allocated
        to the validation set.
    :param random_seed: An integer for seeding the randomness during shuffling.
    :param buffer_size: An integer defining the buffer size for shuffling the dataset.
    :return: A tuple containing the train, validation, and test datasets as
        (train_dataset, val_dataset, test_dataset).
    """
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=random_seed)

    dataset_size = dataset.reduce(0, lambda x, _: x + 1).numpy()

    test_size = int(test_size * dataset_size)
    val_size = int(val_size * dataset_size)
    train_size = dataset_size - test_size - val_size

    train_dataset = dataset.take(train_size)
    rest_dataset = dataset.skip(train_size)

    val_dataset = rest_dataset.take(val_size)
    test_dataset = rest_dataset.skip(val_size)

    return train_dataset, val_dataset, test_dataset
