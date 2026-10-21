"""Transformer encoder-decoder (Vaswani et al., 2017) in Keras.

Post-layer-norm blocks, learned positional embeddings, and token embeddings that can be
initialised from pre-trained FastText vectors and optionally frozen.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size: int, max_len: int, dim: int, weights: np.ndarray | None = None,
                 trainable_tokens: bool = True, **kwargs):
        super().__init__(**kwargs)
        init = tf.keras.initializers.Constant(weights) if weights is not None else "uniform"
        self.tok = layers.Embedding(vocab_size, dim, embeddings_initializer=init, trainable=trainable_tokens)
        self.pos = layers.Embedding(max_len, dim)

    def call(self, x):
        return self.tok(x) + self.pos(tf.range(tf.shape(x)[-1]))


class EncoderBlock(layers.Layer):
    def __init__(self, dim: int, heads: int, ff: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(heads, dim // heads, dropout=dropout)
        self.ffn = tf.keras.Sequential([layers.Dense(ff, activation="relu"), layers.Dense(dim)])
        self.n1, self.n2 = layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)
        self.d1, self.d2 = layers.Dropout(dropout), layers.Dropout(dropout)

    def call(self, x, mask, training=False):
        x = self.n1(x + self.d1(self.att(x, x, attention_mask=mask, training=training), training=training))
        return self.n2(x + self.d2(self.ffn(x), training=training))


class DecoderBlock(layers.Layer):
    def __init__(self, dim: int, heads: int, ff: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.self_att = layers.MultiHeadAttention(heads, dim // heads, dropout=dropout)
        self.cross_att = layers.MultiHeadAttention(heads, dim // heads, dropout=dropout)
        self.ffn = tf.keras.Sequential([layers.Dense(ff, activation="relu"), layers.Dense(dim)])
        self.norms = [layers.LayerNormalization(epsilon=1e-6) for _ in range(3)]
        self.drops = [layers.Dropout(dropout) for _ in range(3)]

    def call(self, y, memory, self_mask, cross_mask, training=False):
        a = self.self_att(y, y, attention_mask=self_mask, training=training)
        y = self.norms[0](y + self.drops[0](a, training=training))
        a = self.cross_att(y, memory, attention_mask=cross_mask, training=training)
        y = self.norms[1](y + self.drops[1](a, training=training))
        return self.norms[2](y + self.drops[2](self.ffn(y), training=training))


def build_transformer(src_vocab: int, tgt_vocab: int, src_len: int, tgt_len: int, *, dim: int, heads: int,
                      ff: int, layers_: int, dropout: float, src_emb=None, tgt_emb=None,
                      trainable_emb: bool = False, pad_id: int = 0) -> tf.keras.Model:
    enc_in = layers.Input((src_len,), dtype="int32", name="encoder_input")
    dec_in = layers.Input((tgt_len,), dtype="int32", name="decoder_input")

    src_pad = layers.Lambda(lambda t: tf.not_equal(t, pad_id)[:, tf.newaxis, :], name="src_mask")(enc_in)
    causal = np.tril(np.ones((tgt_len, tgt_len), dtype=bool))
    self_mask = layers.Lambda(
        lambda t: tf.logical_and(causal[tf.newaxis], tf.not_equal(t, pad_id)[:, tf.newaxis, :]),
        name="causal_mask")(dec_in)

    x = TokenAndPositionEmbedding(src_vocab, src_len, dim, src_emb, trainable_emb, name="src_embed")(enc_in)
    for i in range(layers_):
        x = EncoderBlock(dim, heads, ff, dropout, name=f"encoder_{i}")(x, src_pad)
    y = TokenAndPositionEmbedding(tgt_vocab, tgt_len, dim, tgt_emb, trainable_emb, name="tgt_embed")(dec_in)
    for i in range(layers_):
        y = DecoderBlock(dim, heads, ff, dropout, name=f"decoder_{i}")(y, x, self_mask, src_pad)
    logits = layers.Dense(tgt_vocab, name="logits")(layers.Dropout(dropout)(y))
    return tf.keras.Model([enc_in, dec_in], logits, name="transformer_nmt")


def masked_loss(pad_id: int = 0):
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
        return tf.reduce_sum(ce(y_true, y_pred) * mask) / tf.reduce_sum(mask)

    return loss


def masked_accuracy(pad_id: int = 0):
    def acc(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
        hit = tf.cast(tf.equal(y_true, tf.argmax(y_pred, -1, output_type=tf.int32)), tf.float32)
        return tf.reduce_sum(hit * mask) / tf.reduce_sum(mask)

    acc.__name__ = "token_accuracy"
    return acc


def greedy_decode(model: tf.keras.Model, enc: np.ndarray, tgt_len: int, sos: int, eos: int, pad_id: int = 0):
    """Batched greedy decoding. ``enc`` has shape (batch, src_len)."""
    dec = np.full((len(enc), tgt_len), pad_id, dtype=np.int32)
    dec[:, 0] = sos
    done = np.zeros(len(enc), dtype=bool)
    for t in range(1, tgt_len):
        logits = model([enc, dec], training=False)
        nxt = tf.argmax(logits[:, t - 1], -1, output_type=tf.int32).numpy()
        dec[:, t] = np.where(done, pad_id, nxt)
        done |= nxt == eos
        if done.all():
            break
    return dec[:, 1:]
