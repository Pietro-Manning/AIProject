import tensorflow as tf

from models.model_layers import TokenAndPositionEmbedding, TransformerBlock
from config.config_loader import CONFIG

def transformer_model(maxlen: int = CONFIG["max_seq_length"], vocab_size: int = CONFIG["vocab_size"],
                      embedding_dim: int = 128, num_heads: int = 8,
                      dropping_rate: float = 0.1, num_classes: int = 130, use_bert: bool = True) -> tf.keras.Model:

    if use_bert:
        inputs = tf.keras.Input(shape=(maxlen,768), dtype=tf.float32, name="bert_embeddings")
    else:
        inputs = tf.keras.Input(shape=(maxlen,), dtype=tf.int32, name="input_ids")

    x0 = TokenAndPositionEmbedding(maxlen=maxlen, vocab_size=vocab_size, embedding_dim=embedding_dim, use_bert=use_bert)(inputs)

    # First transformer block
    t1 = TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads, dropping_rate=dropping_rate,
                          use_bert_embedding=CONFIG["use_bert_embedding"], name="transformer_block_1")(x0)

    # Second transformer block with skip connection
    t2 = TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads, dropping_rate=dropping_rate,
                          use_bert_embedding=CONFIG["use_bert_embedding"], name="transformer_block_2")(x0 + t1)

    # Convolutional block
    conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same", name="conv1d_1")(t1 + t2)
    conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation="relu", padding="same", name="conv1d_2")(conv1)

    # Aggregation block
    output = tf.keras.layers.GlobalAveragePooling1D()(conv2)
    output = tf.keras.layers.Dropout(dropping_rate)(output)
    output = tf.keras.layers.Dense(64, activation="relu")(output)
    output = tf.keras.layers.Dropout(dropping_rate)(output)

    # Classification block
    classifier = tf.keras.layers.Dense(num_classes, activation="sigmoid")(output)

    return tf.keras.models.Model(inputs=inputs, outputs=classifier, name="multilabel_classifier")
