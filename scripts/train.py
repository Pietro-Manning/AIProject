
import tensorflow as tf
import tensorflow_text as text
import time
import utils.utils as utils

from models.model_builder import transformer_model
from src.utils.tokenizer_utils import tokenize_dataset
from config.config_loader import CONFIG
from src.data.prepare_data import split_dataset, get_and_preprocess_mnds

# getting global configurations
BATCH_SIZE = int(CONFIG["batch_size"])
SHUFFLE_BUFFER_SIZE = int(CONFIG["shuffle_buffer_size"])
LEARNING_RATE = float(CONFIG["learning_rate"])
EPOCHS = int(CONFIG["epochs"])
MAX_SEQ_LENGTH = int(CONFIG["max_seq_length"])
USE_BERT_EMBEDDING = bool(CONFIG["use_bert_embedding"])
EMBEDDING_DIM = int(CONFIG["embedding_dim"])
VOCAB_SIZE=int(CONFIG["vocab_size"])
NUM_HEAD= CONFIG["num_head"]

def get_and_train_model(data_size = None):
    """
    Gets the data, preprocesses it, splits it into datasets, trains a text
    classification model using Transformer architecture, evaluates the model,
    and logs the necessary details for further analysis.

    This function supports both BERT-based embeddings and standard embeddings
    depending on a predefined setting. It also visualizes the TensorFlow graph
    in TensorBoard, implements a series of callbacks for better training
    stability, and collects evaluation metrics on the test dataset.

    :param data_size: The number of data samples to be used for training,
                      validation, and testing.
    :type data_size: int
    :return: A tuple containing the training history object, the run number of the
             model, the log directory path, the sizes of the training, validation,
             and test datasets, and the evaluation metrics on the test dataset.
    :rtype: Tuple[tf.keras.callbacks.History, int, str, int, int, int, Dict[str, Any]]
    """
    log_dir = utils.get_next_run_dir()
    dataset, classes = get_and_preprocess_mnds(data_size)
    train_dataset, validation_dataset, test_dataset = split_dataset(dataset)

    train_count = tf.data.experimental.cardinality(train_dataset).numpy()
    validation_count = tf.data.experimental.cardinality(validation_dataset).numpy()
    test_count = tf.data.experimental.cardinality(test_dataset).numpy()

    filepath, run_number = utils.retrieve_next_model_filename()

    # choosing from bert embedding or standard embedding
    if USE_BERT_EMBEDDING:
        train_dataset = (utils.preprocess_with_bert_as_embedder(train_dataset).shuffle(SHUFFLE_BUFFER_SIZE)
                         .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

        val_dataset = utils.preprocess_with_bert_as_embedder(validation_dataset).batch(BATCH_SIZE)
        tst_dataset = utils.preprocess_with_bert_as_embedder(test_dataset).batch(BATCH_SIZE)

    else:
        train_dataset, bert_tokenizer = tokenize_dataset(train_dataset)
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_dataset, _ = tokenize_dataset(validation_dataset)
        val_dataset = val_dataset.batch(BATCH_SIZE)

        tst_dataset, _ = tokenize_dataset(test_dataset)
        tst_dataset = tst_dataset.batch(BATCH_SIZE)

        del bert_tokenizer

    model = transformer_model(
        num_classes=len(classes),
        vocab_size=VOCAB_SIZE,
        use_bert=USE_BERT_EMBEDDING,
        embedding_dim = EMBEDDING_DIM,
        num_heads=NUM_HEAD
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(multi_label=True),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryAccuracy()
        ]
    )

    # creating the tensorflow graph so that it can be visualized in the tensorboard dashboard
    utils.get_tb_graph(log_dir, model, USE_BERT_EMBEDDING, VOCAB_SIZE)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="batch", write_images=True, write_graph=True,
                                       histogram_freq=1, profile_batch=0)
    ]

    history = model.fit(train_dataset,validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)
    test_metrics_names = ["test_loss", "test_auc", "test_precision", "test_recall", "test_binary_accuracy"]
    results = model.evaluate(tst_dataset)
    metrics_on_test_dataset = dict(zip(test_metrics_names, results))
    utils.write_classes_to_file(classes, filepath)

    return history, run_number, log_dir, train_count, validation_count, test_count, metrics_on_test_dataset

def start_training():
    """
    Starts the training process for a machine learning model, measures and formats the
    training duration, and generates a description of the model, including key training
    metrics and results.

    :return: None
    """
    start_time = time.time()
    history, run_number, log_dir, train_count, validation_count, test_count, metrics_on_test_dataset = get_and_train_model()
    training_time = time.time() - start_time

    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)

    formatted_training_time = f"{hours} hours, {minutes} minutes and {seconds} seconds"

    utils.generate_description_model_file(run_number,
                                     "template.txt",
                                     f"{log_dir}/model_description.txt", history,
                                          formatted_training_time, train_count, validation_count,
                                          test_count, metrics_on_test_dataset)
# start the training
start_training()