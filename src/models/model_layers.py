import tensorflow as tf
from config.config_loader import CONFIG
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="Custom", name="MultiHeadSelfAttention")
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Multi-head self-attention layer.

    This class defines a multi-head self-attention layer used for capturing dependencies
    between different positions in input sequences. It leverages the `MultiHeadAttention`
    layer from TensorFlow Keras to enable simultaneous attention mechanisms (multi-head).
    It is primarily used in models for sequence processing tasks, such as transformers.

    :ivar attention: The underlying TensorFlow multi-head attention layer that conducts
        attention operations.
    :type attention: tf.keras.layers.MultiHeadAttention
    """
    def __init__(self, embedding_dim, num_heads=8):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

    def call(self, inputs):
        attn_output = self.attention(inputs, inputs)
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="Custom", name="TokenAndPositionEmbedding")
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """
    A custom TensorFlow Keras Layer for token and position embedding.

    This class provides an implementation for token embedding and positional
    embedding, either using a pre-calculated BERT embedding or a classic
    trainable embedding layer. It is designed to handle input sequences and
    apply the necessary embeddings, facilitating their use in NLP tasks or
    other sequence-based models.

    :ivar use_bert: Whether to use BERT embeddings (pre-calculated) or classic
        trainable embeddings.
    :type use_bert: bool
    :ivar maxlen: Maximum sequence length that the layer expects inputs to have.
    :type maxlen: int
    :ivar embedding_token: Token embedding layer for classic trainable embedding.
        This is only initialized if `use_bert` is False.
    :type embedding_token: tf.keras.layers.Embedding
    :ivar positional_embedding_layer: Positional embedding layer for classic
        embedding. This is only initialized if `use_bert` is False.
    :type positional_embedding_layer: tf.keras.layers.Embedding
    """
    def __init__(self, maxlen:int, vocab_size:int, embedding_dim:int, use_bert:bool = True, name:str = "embedding_layer", **kwargs):

        super().__init__(name=name, **kwargs)

        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_bert = use_bert
        self.name = name


        if not self.use_bert:
            # classic embedding
            self.embedding_token = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, name=f"{name}_token")
            self.positional_embedding_layer = tf.keras.layers.Embedding(input_dim=self.maxlen, output_dim=self.embedding_dim, name=f"{name}_positional")


    def call(self, input_ids, training:bool=False):

        if self.use_bert:
            # When using BERT, embeddings must be already calculated and presented as input
            return input_ids

        else:
            # base model embedding
            positions = tf.range(start=0, limit=self.maxlen, delta=1)
            positions = self.positional_embedding_layer(positions)
            token_embeddings = self.embedding_token(input_ids)
            return token_embeddings + positions

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'use_bert': self.use_bert,
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="Custom", name="TransformerBlock")
class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embedding_dim:int, num_heads:int, dropping_rate:float=0.1,
                 use_bert_embedding:bool=True, name:str= "transformer_block", **kwargs):
        super().__init__(name=name, **kwargs)

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropping_rate = dropping_rate
        self.use_bert_embedding = use_bert_embedding
        self.ff_dim = CONFIG["ff_dim_bert"] if use_bert_embedding else CONFIG["ff_dim_standard"]

        self.att = MultiHeadSelfAttention(768 if use_bert_embedding else embedding_dim, num_heads)
        self.ffn = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation="relu"),
            tf.keras.layers.Dropout(dropping_rate),
            tf.keras.layers.Dense(768 if use_bert_embedding else embedding_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=float(CONFIG["normalization"]))
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=float(CONFIG["normalization"]))
        self.dropout1 = tf.keras.layers.Dropout(dropping_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropping_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        residual1 = inputs + attn_output
        out1 = self.layernorm1(residual1)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        residual2 = out1 + ffn_output

        return self.layernorm2(residual2)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'dropping_rate': self.dropping_rate,
            'use_bert_embedding': self.use_bert_embedding
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
