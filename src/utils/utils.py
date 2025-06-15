import os
from pathlib import Path
from config.config_loader import CONFIG

def get_src_path():
    """
    Returns the source directory path of the current script.

    This function determines the absolute path to the directory that
    is two levels above the current script location (the parent of the
    parent directory). This is useful when needing to construct paths
    relative to the project's source directory.

    :return: The absolute path of the parent directory is two levels above
             the current script.
    :rtype: Path
    """
    return Path(__file__).resolve().parent.parent

def get_log_dir():
    """
    Determines and returns the directory path for storing logs used by
    TensorBoard. The path is constructed based on the embedding type
    defined in the configuration. If the directory does not exist, it
    is automatically created.

    :return: The absolute path of the logs directory.
    :rtype: str
    """
    is_bert_embedding = CONFIG["use_bert_embedding"]
    src_path = get_src_path()
    log_dir = os.path.join(src_path, "logs", "tensorboard")

    if is_bert_embedding: log_dir = os.path.join(log_dir, "bert_embedding")
    else: log_dir = os.path.join(log_dir, "standard_embedding")

    if not os.path.exists(log_dir): os.makedirs(log_dir)

    return log_dir


def get_next_run_dir():
    """
    Determines the next unique directory for logging or output storage.

    The function calculates the next available directory name for a run by identifying
    existing run directories and incrementing the highest run number. It ensures
    directory creation for storage in the appropriate path.

    :return: Absolute path to the newly created run directory.
    :rtype: str
    """
    log_dir = get_log_dir()

    base_log_dir = os.path.dirname(log_dir)
    max_run_number = -1

    for _dir in os.listdir(base_log_dir):
        existing_runs = [d for d in os.listdir(os.path.join(base_log_dir, _dir)) if os.path.isdir(os.path.join(base_log_dir, _dir, d)) and d.startswith('run_')]
        run_number = max([int(d.split('_')[1]) for d in existing_runs if d.split('_')[1]], default=0) + 1
        if run_number > max_run_number: max_run_number = run_number

    run_dir = os.path.join(log_dir, f"run_{max_run_number}")
    os.makedirs(run_dir)
    return run_dir

def retrieve_next_model_filename():

    use_bert_embedding = CONFIG["use_bert_embedding"]
    base_model_dir = os.path.join(get_src_path(), "models", "data")

    if use_bert_embedding: base_model_dir = os.path.join(base_model_dir, "bert_embedding")
    else: base_model_dir = os.path.join(base_model_dir, "standard_embedding")

    if not os.path.exists(base_model_dir):
        os.makedirs(base_model_dir)

    root_base_model_dir = os.path.dirname(base_model_dir)
    existing_models = []

    for _dir in os.listdir(root_base_model_dir):
        for f in os.listdir(os.path.join(root_base_model_dir, _dir)):
            if f.startswith("run_"):
                existing_models.append(f)

    model_numbers = [int(f.split("_")[1]) for f in existing_models]
    next_model_number = max(model_numbers, default=0) + 1

    model_dir = os.path.join(base_model_dir, f"run_{next_model_number}")
    os.makedirs(model_dir)

    return os.path.join(model_dir, f"model_run_{next_model_number}.keras"), next_model_number

def get_tb_graph(log_dir, model, use_bert_embedding, vocab_size):
    """
    Generates a TensorBoard graph representation of a TensorFlow model. This function takes
    a specified model and a log directory where the graph will be saved. It allows for
    dynamic handling of BERT embeddings or regular tokenized sequences depending on the
    `use_bert_embedding` flag. The function validates inputs and freezes the computational
    graph before logging to TensorBoard.

    :param log_dir: The directory in which to save the TensorBoard graph. Must exist before
            calling this function.
    :type log_dir: str
    :param model: The TensorFlow model to visualize in TensorBoard. Must be an instance of
        `tf.keras.Model` and callable.
    :type model: tf.keras.Model
    :param use_bert_embedding: A boolean flag indicating if the model requires BERT-style
        embeddings. If True, the input tensor shape corresponds to BERT embeddings; otherwise,
        it assumes tokenized sequences.
    :type use_bert_embedding: bool
    :param vocab_size: The size of the vocabulary for tokenized input. This parameter is
        used only when `use_bert_embedding` is False.
    :type vocab_size: int
    :return: None
    :rtype: None
    :raises FileNotFoundError: If the `log_dir` does not exist.
    :raises TypeError: If the `model` is not an instance of `tf.keras.Model`.
    :raises ValueError: If the `model` is not callable.
    """
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from config.config_loader import CONFIG

    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"The specified directory doesn't exist: {log_dir}")

    if not isinstance(model, tf.keras.Model):
        raise TypeError("Model parameter must be a tf.keras.Model instance")

    if not callable(model):
        raise ValueError("Model is not callable")


    max_seq_length = int(CONFIG["max_seq_length"])

    if use_bert_embedding:
        dummy_input = tf.random.uniform((1, max_seq_length, 768), dtype=tf.float32)
        model(dummy_input)
        concrete_func = tf.function(model).get_concrete_function(tf.TensorSpec([1, max_seq_length, 768], tf.float32))
    else:
        dummy_input = tf.random.uniform((1, max_seq_length), dtype=tf.int32, minval=0, maxval=vocab_size)
        model(dummy_input)
        concrete_func = tf.function(model).get_concrete_function(tf.TensorSpec([1, max_seq_length], tf.float32))

    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.graph(graph_def)

def generate_description_model_file(run_number, template_file_name, output_path, history, training_time, train_count,
                                    validation_count, test_count, metrics_on_test_dataset):
    """
    Generate and write a description file for the trained model using a provided template.

    This function reads a template file, formats it with specified data from the training
    history, configuration values, and additional metrics, and writes the formatted content
    to a specified output path. It serves as a mechanism for dynamically generating
    descriptive files that summarize training processes and results.

    :param run_number: The identifier for the training run corresponding to this model.
    :type run_number: int
    :param template_file_name: The name of the template file used for generating the description.
    :type template_file_name: str
    :param output_path: The file path where the generated description file will be saved.
    :type output_path: str
    :param history: The training history containing lists or arrays of performance metrics
                    (accuracy, loss, etc.) per epoch.
    :type history: tensorflow.keras.callbacks.History
    :param training_time: The total time taken for training the model (in seconds or other units).
    :type training_time: str
    :param train_count: The total number of examples in the training dataset.
    :type train_count: int
    :param validation_count: The total number of examples in the validation dataset.
    :type validation_count: int
    :param test_count: The total number of examples in the test dataset.
    :type test_count: int
    :param metrics_on_test_dataset: A dictionary containing performance metrics of the model
                                     evaluated on the test dataset.
    :type metrics_on_test_dataset: dict
    :return: None
    """
    absolut_template_path = f"{get_src_path().parent}/templates/{template_file_name}"
    with open(absolut_template_path, 'r') as template_file:
        template_content = template_file.read()

    training_data = {
        "final_training_loss": history.history["loss"][-1],
        "final_validation_loss": history.history["val_loss"][-1],
        "final_training_auc": history.history["auc"][-1],
        "final_validation_auc": history.history['val_auc'][-1],
        "final_training_precision": history.history["precision"][-1],
        "final_validation_precision": history.history['val_precision'][-1],
        "final_training_recall": history.history["recall"][-1],
        "final_validation_recall": history.history['val_recall'][-1],
        "final_training_binary_accuracy": history.history["binary_accuracy"][-1],
        "final_validation_binary_accuracy": history.history['val_binary_accuracy'][-1],
        "epochs_trained": len(history.history["loss"]),
        "training_time": training_time,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count
    }

    result_content = template_content.format(num_experiment=run_number, **CONFIG,
                                             **training_data, **metrics_on_test_dataset)

    with open(output_path, 'w') as output_file:
        output_file.write(result_content)

def preprocess_with_bert_as_embedder(dataset):
    """
    Preprocesses a given dataset using TensorFlow Hub's BERT preprocessing and embedding
    models. The function extracts embeddings from a BERT model for each text in the
    dataset and collects the processed embeddings along with their corresponding labels
    into a new TensorFlow dataset.

    :param dataset: A TensorFlow Dataset where each element is a tuple containing
        a piece of text and its associated label.
    :type dataset: tf.data.Dataset

    :return: A TensorFlow Dataset containing preprocessed embeddings and their
        corresponding labels, where the embeddings are a stack of processed output
        from the BERT model.
    :rtype: tf.data.Dataset
    """

    import tensorflow_hub as hub
    import tensorflow as tf


    bert_preprocess_model = hub.KerasLayer(CONFIG["bert_preprocess_model"])
    bert_model = hub.KerasLayer(CONFIG["bert_model"], trainable=False)

    preprocessed_texts = []
    labels = []

    for text, label in dataset.as_numpy_iterator():
        input_text = tf.convert_to_tensor([text])  # shape (1,)
        preprocessed = bert_preprocess_model(input_text)
        embedding = bert_model(preprocessed)["sequence_output"][0]  # shape (128, 768)
        preprocessed_texts.append(embedding)
        labels.append(label)

    final_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.stack(preprocessed_texts), tf.convert_to_tensor(labels, dtype=tf.float32)))

    return final_dataset

def preprocess_with_bert_as_embedder_4_inference(dataset):
    """
    Preprocess text data using BERT as an embedder for inference. This function
    uses TensorFlow Hub's BERT preprocess and BERT model to generate sequence
    embeddings for each input text sample in the dataset. Each text sample is
    preprocessed, passed through the BERT model, and transformed into its
    corresponding embedding. The preprocessed embeddings are collected and returned
    as a TensorFlow dataset.

    :param dataset: TensorFlow data.Dataset object containing the input text
        samples to be processed.
    :type dataset: tensorflow.data.Dataset

    :return: A TensorFlow dataset containing the preprocessed and embedded text
        samples as BERT sequence embeddings.
    :rtype: tensorflow.data.Dataset
    """
    import tensorflow_hub as hub
    import tensorflow as tf
    import tensorflow_text as text

    bert_preprocess_model = hub.KerasLayer(CONFIG["bert_preprocess_model"])
    bert_model = hub.KerasLayer(CONFIG["bert_model"], trainable=False)

    preprocessed_texts = []

    for text in dataset.as_numpy_iterator():
        input_text = tf.convert_to_tensor([text])  # shape (1,)
        preprocessed = bert_preprocess_model(input_text)
        embedding = bert_model(preprocessed)["sequence_output"][0]  # shape (128, 768)
        preprocessed_texts.append(embedding)

    final_dataset = tf.data.Dataset.from_tensor_slices(tf.stack(preprocessed_texts))

    return final_dataset

def write_classes_to_file(classes, filepath):
    """
    Writes a list of class names to a file named "classes.txt" in the directory
    containing the provided file path.

    This function takes a list of class names and creates a new file named
    "classes.txt" in the same directory as the specified `filepath`. It writes each
    class name to the file on a new line.

    :param classes: List of class names to be written to the file.
    :type classes: list of str
    :param filepath: The path of the file whose directory will be used to create
        "classes.txt".
    :type filepath: str
    :return: None
    """
    classes_file = os.path.join(os.path.dirname(filepath), "classes.txt")
    with open(classes_file, "w") as file:
        for class_name in classes:
            file.write(class_name + "\n")


def get_class_names(class_path):
    """
    Retrieves class names from a file named ``classes.txt`` located in the same
    directory as the provided ``class_path``. The function reads all lines from
    the file and returns a list of stripped lines, each representing a class name.

    :param class_path: The path of the file for determining the directory where
        ``classes.txt`` is located.
    :type class_path: str
    :return: A list of class names read from the ``classes.txt`` file.
    :rtype: list[str]
    """
    path = os.path.join(os.path.dirname(class_path), "classes.txt")
    with open(path, "r") as path:
        class_names = [line.strip() for line in path]
    return class_names