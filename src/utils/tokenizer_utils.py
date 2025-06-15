import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from config.config_loader import CONFIG

BATCH_SIZE = CONFIG["batch_size"]
MAX_SEQ_LENGTH = CONFIG["max_seq_length"]

def tokenize_dataset(dataset, is_training=True):
    """
    This function processes a given dataset using a pre-trained BERT preprocessing model, generating
    tokenized text that can be used as input for further machine learning tasks. Depending on the
    value of the `is_training` parameter, it either includes labels (for training) or excludes them
    (for inference). The function returns the processed dataset and the preprocessing model used.

    :param dataset: A dataset containing text data (and optional labels if `is_training` is True) that
        will be tokenized using a pre-trained BERT preprocessing model.
        (Expected format: tf.data.Dataset with elements as either tuples (for training) or strings
        (for inference)).

    :param is_training: A boolean flag. If True, the dataset is assumed to include labels, and these
        labels will also be part of the returned processed dataset; otherwise, only tokenized text will
        be returned.

    :return: A tuple containing:
        - The preprocessed dataset (either with input_texts and labels, or input_texts only),
        as `tf.data.Dataset`.
        - The KerasLayer instance of the BERT preprocessing model used for processing text.
    """
    bert_preprocess_model = hub.KerasLayer(CONFIG["bert_preprocess_model"])

    preprocessed_texts = []

    if is_training:

        labels = []
        for _text, label in dataset.as_numpy_iterator():
            input_text = tf.convert_to_tensor([_text])
            preprocessed_text = bert_preprocess_model(input_text)
            preprocessed_texts.append(tf.squeeze(preprocessed_text["input_word_ids"], axis=0))
            labels.append(label)

        final_dataset = tf.data.Dataset.from_tensor_slices(
            (tf.stack(preprocessed_texts), tf.convert_to_tensor(labels, dtype=tf.float32)))

    else:

        for _text in dataset.as_numpy_iterator():
            input_text = tf.convert_to_tensor([_text])
            preprocessed_text = bert_preprocess_model(input_text)
            preprocessed_texts.append(tf.squeeze(preprocessed_text["input_word_ids"], axis=0))

        final_dataset = tf.data.Dataset.from_tensor_slices(tf.stack(preprocessed_texts))

    return final_dataset, bert_preprocess_model





