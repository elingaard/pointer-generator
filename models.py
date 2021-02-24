import tensorflow as tf
from tensorflow import keras

class Encoder(keras.Model):
    """Bi-directional GRU encoder"""
    def __init__(self, vocab_size, embedding_dim, hidden_units,embedding_matrix):
        super().__init__()

        self.hidden_units = hidden_units
        if embedding_matrix is not None:
            self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])
        else:
            self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.bi_gru = keras.layers.Bidirectional(keras.layers.GRU(
                hidden_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform',
            ))

    def call(self,encoder_input,encoder_states):
        """Forward pass of encoder
        Args:
        encoder_input(int tensor: (batch_size,seq_length) ): sequence(s) of vocabulary ids
        encoder_states(list, len=2): encoder forward and backward state

        Returns:
        encoder_output(float tensor: (batch_size,seq_length,hidden_dim) ): encoded space of each sequence
        encoder_states(list, len=2): updated encoder states
        """

        encoder_emb = self.embedding(encoder_input)
        encoder_output, state_fwd, state_back = self.bi_gru(encoder_emb,initial_state=encoder_states)
        encoder_states = [state_fwd,state_back]

        return encoder_output, encoder_states

class BahdanauAttention(keras.Model):
    """Attention layer as described in: Neural Machine Translation by Jointly Learning to Align and Translate"""
    def __init__(self, hidden_units,is_coverage=False):
        super().__init__()

        self.Wh = keras.layers.Dense(hidden_units) # weight matrix for encoder hidden state
        self.Ws = keras.layers.Dense(hidden_units) # weight matrix for decoder state
        self.V = keras.layers.Dense(1)
        self.coverage = is_coverage
        if self.coverage is False:
            self.wc = keras.layers.Dense(1,kernel_initializer='zeros') # weight vector for coverage
            self.wc.trainable = False
        else:
            self.wc = keras.layers.Dense(1)

    def call(self, decoder_state, encoder_output,coverage_vector):
        """Forward pass of attention layer
        Args:
        decoder_state(float tensor: (batch_size,hidden_dim) )
        encoder_output(float tensor: (batch_size,seq_length,hidden_dim) )
        coverage_vector(float tensor: (batch_size,seq_length) )

        Returns:
        context_vector(float tensor: (batch_size,hidden_dim) )
        attention_weights(float tensor: (batch_size,seq_length) )
        coverage_vector(float tensor: (batch_size,seq_length) )
        """

        # calculate attention scores
        decoder_state = tf.expand_dims(decoder_state, 1)
        coverage_vector = tf.expand_dims(coverage_vector, 1)
        score = self.V(tf.nn.tanh(
                        self.Wh(encoder_output) +
                        self.Ws(decoder_state) +
                        self.wc(coverage_vector)
                        ))

        attention_weights = tf.nn.softmax(score, axis=1)
        coverage_vector = tf.squeeze(coverage_vector,1)
        if self.coverage is True:
          coverage_vector+=tf.squeeze(attention_weights)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights, coverage_vector

class Decoder(keras.Model):
    """Bi-directional GRU decoder with two dense layers in the end to model the vocabulary distribution"""
    def __init__(self, vocab_size, embedding_dim, hidden_units,embedding_matrix):
        super().__init__()

        self.hidden_units = hidden_units
        if embedding_matrix is not None:
            self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])
        else:
            self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(
            hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
        )
        self.W1 = keras.layers.Dense(hidden_units)
        self.W2 = keras.layers.Dense(vocab_size)


    def call(self, decoder_input, decoder_state, encoder_output,context_vector):
        """Forward pass of decoder

        Args:
        decoder_input(int tensor: (batch_size,1) )
        decoder_state(float tensor: (batch_size,hidden_dim) )
        encoder_output(float tensor: (batch_size,seq_length,hidden_dim) )
        coverage_vector(float tensor: (batch_size,seq_length))

        Returns:
        p_vocab(float tensor: (batch_size,vocab_size) )
        decoder_state(float tensor: (batch_size,hidden_dim) )
        """

        decoder_emb = self.embedding(decoder_input) # (batch_size, seq_length, hidden_units)
        decoder_output , decoder_state = self.gru(decoder_emb,initial_state=decoder_state)
        concat_vector = tf.concat([context_vector,decoder_state], axis=-1)
        concat_vector = tf.reshape(concat_vector, (-1, concat_vector.shape[1]))
        p_vocab = tf.nn.log_softmax(self.W2(self.W1(concat_vector)))

        return p_vocab, decoder_state






