import os
import re
import tensorflow as tf

from config.config_loader import CONFIG
from models.model_layers import MultiHeadSelfAttention, TokenAndPositionEmbedding, TransformerBlock
from utils import utils, tokenizer_utils

MODEL_PATH = CONFIG["model_path_4_inference"]
THRESHOLD = float(CONFIG["threshold"])

IS_BERT_USED = True if os.path.basename(os.path.dirname(os.path.dirname(MODEL_PATH))) == "bert_embedding" else False

def load_paragraphs():
    """
    Function to load and parse paragraphs from a text file specified by its path.

    This function reads content from the file "input_to_infer.txt" located in the
    same directory as the defined `MODEL_PATH`. It splits the content into paragraphs
    based on double newlines or multiple consecutive newlines. If the file does not
    exist, an error will be raised.

    :raises FileNotFoundError: If the specified file does not exist.
    :return: A list of paragraphs extracted from the text file.
    :rtype: list[str]
    """

    file_path = os.path.join(os.path.dirname(MODEL_PATH), "input_to_infer.txt")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"File '{file_path}' non found. Make sure the path and file name are corrects!")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = re.split(r"\n\s*\n", content.strip())

    return paragraphs


def preprocess_paragraphs(dataset_paragraphs: tf.data.Dataset):
    """
    Preprocess dataset row_paragraphs based on the configuration and preprocessors available.
    The function applies a preprocessing pipeline that varies depending on whether BERT
    is used. If BERT is enabled, it uses a BERT-based embedding preprocessing method
    for inference purposes. If BERT is not used, it alternatively tokenizes the given dataset
    using a tokenizer utility. The processed dataset is then batched according to the
    configured batch size.

    :param dataset_paragraphs:
        A TensorFlow dataset containing row_paragraphs to preprocess. The preprocessor type
        depends on the BERT configuration.
    :return:
        A batched TensorFlow dataset after applying the preprocessing pipeline.
    """
    if IS_BERT_USED: final_dataset_paragraphs = utils.preprocess_with_bert_as_embedder_4_inference(dataset_paragraphs)
    else: final_dataset_paragraphs, _ = tokenizer_utils.tokenize_dataset(dataset_paragraphs, is_training=False)
    return final_dataset_paragraphs.batch(CONFIG["batch_size"])


def load_model():
    """
    Loads a TensorFlow Keras model from the specified file path.

    This function uses TensorFlow's Keras API to load a pre-trained
    model stored in the file system at the location defined by the
    constant `MODEL_PATH`. This is commonly used to restore and use
    a previously saved model for inference or fine-tuning.

    :raises OSError: If the file does not exist or cannot be loaded
        as a valid TensorFlow Keras model.

    :return: The loaded TensorFlow Keras model.
    :rtype: tf.keras.models.Model
    """
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects = {
            "MultiHeadSelfAttention": MultiHeadSelfAttention,
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock
        }
    )


def predict_and_write(model, row_paragraphs, classes, threshold=0.50):
    """
    Predicts classes for given paragraphs using a deep learning model and writes the results to a file
    in a formatted manner. The function preprocesses the input paragraphs, performs inference using
    the specified model, and outputs predictions for each paragraph with associated classes.

    :param model: The deep learning model used for prediction.
    :type model: tf.keras.Model
    :param row_paragraphs: List of raw paragraphs to be processed and classified.
    :type row_paragraphs: list[str]
    :param classes: List of class names corresponding to the model's output indices.
    :type classes: list[str]
    :param threshold: The confidence threshold for class prediction. Defaults to 0.50.
    :type threshold: float, optional
    :return: None
    """
    base_run_directory = os.path.dirname(CONFIG["model_path_4_inference"])
    output_path = os.path.join(base_run_directory, "output_inferred.txt")

    paragraphs = [p.replace("\n", " ").strip() for p in row_paragraphs]

    dataset_paragraphs = tf.data.Dataset.from_tensor_slices(paragraphs)
    tokenized_dataset = preprocess_paragraphs(dataset_paragraphs)
    predictions = model.predict(tokenized_dataset)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for paragraph, probs in zip(row_paragraphs, predictions):
            sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            top_k = [(i, p) for i, p in sorted_probs[:5] if p >= threshold]

            class_names = [f"{classes[i]}: {prob * 100:.1f}%" for i, prob in top_k]
            class_names = "\n==> ".join(class_names)

            out_f.write(f"{paragraph}\n\nPredicted class/classes: "
                        f"\n\n==> {class_names}\n\n------------------------------------"
                        f"--------------------------------\n")

    print(f"Inference completed. Output written to {output_path}")

def do_inference():
    """
    Performs inference by loading paragraphs, a pre-trained model, and the associated
    classes, then making predictions and writing the outputs for further use.
    This function serves as the entry point for processing and classifying input data
    using the loaded machine learning model and its configuration.

    :return: None
    """
    row_paragraphs = load_paragraphs()
    model = load_model()

    classes = utils.get_class_names(class_path=MODEL_PATH)

    predict_and_write(model, row_paragraphs, classes, threshold=THRESHOLD)


do_inference()