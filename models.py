import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units,embedding_matrix):
        super().__init__()

        self.hidden_units = hidden_units
        if embedding_matrix is not None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.bi_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
        ))

    def call(self, encoder_input,encoder_states):
        # inputs: encoder_input = (batch_size, seq_length)
        #         encoder_states = list[(batch_size, hidden_units),(batch_size, hidden_units)]

        # embedding look-up layer
        encoder_emb = self.embedding(encoder_input) # (batch_size,seq_length,embedding_dim)

        # encoder_output = (batch_size,seq_length,hidden_units)
        # encoder_states = (batch_size,hidden_units)
        encoder_output, state_fwd, state_back = self.bi_gru(encoder_emb,initial_state=encoder_states)
        encoder_states = [state_fwd,state_back]

        return encoder_output, encoder_states

class BahdanauAttention(tf.keras.Model):
    def __init__(self, hidden_units,is_coverage=False):
        super().__init__()

        self.Wh = tf.keras.layers.Dense(hidden_units) # weight matrix for encoder hidden state
        self.Ws = tf.keras.layers.Dense(hidden_units) # weight matrix for decoder state
        self.V = tf.keras.layers.Dense(1)
        self.coverage = is_coverage
        if self.coverage is False:
            self.wc = tf.keras.layers.Dense(1,kernel_initializer='zeros') # weight vector for coverage
            self.wc.trainable = False
        else:
            self.wc = tf.keras.layers.Dense(1)

    def call(self, decoder_state, encoder_output,coverage_vector):
        # inputs: decoder_state = (batch_size, hidden_units)
        #         encoder_output = (batch_size, seq_length, hidden_units)
        #         coverage_vector = (batch_size, seq_length)

        # expand dimension of decoder state and coverage vector to allow addition
        decoder_state = tf.expand_dims(decoder_state, 1) # (batch_size, 1, hidden_units)
        coverage_vector = tf.expand_dims(coverage_vector, 1) # (batch_size, 1, seq_length)

        # calculate attention scores
        # score = (batch_size, length, 1)
        score = self.V(tf.nn.tanh(
                        self.Wh(encoder_output) +  # (batch_size, length, hidden_units) -> (batch_size, length, attention_units)
                        self.Ws(decoder_state) +  # (batch_size, 1, hidden_units) -> (batch_size, 1, attention_units)
                        self.wc(coverage_vector) # (batch_size, 1, seq_length) -> (batch_size, 1, 1)
                        ))

        attention_weights = tf.nn.softmax(score, axis=1) # (batch_size, seq_length, 1)
        # only update coverage vector if coverage is enabled
        coverage_vector = tf.squeeze(coverage_vector,1) # (batch_size, seq_length)
        if self.coverage is True:
          coverage_vector+=tf.squeeze(attention_weights)

        context_vector = attention_weights * encoder_output # (batch_size, seq_length, hidden_units)
        context_vector = tf.reduce_sum(context_vector, axis=1) # (batch_size, hidden_units)

        return context_vector, attention_weights, coverage_vector

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units,embedding_matrix):
        super().__init__()

        self.hidden_units = hidden_units
        if embedding_matrix is not None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
        )
        self.W1 = tf.keras.layers.Dense(hidden_units)
        self.W2 = tf.keras.layers.Dense(vocab_size)


    def call(self, decoder_input, decoder_state, encoder_output,context_vector):
        # inputs: decoder_input = (batch_size, 1)
        #         decoder_state = (batch_size, hidden_units)
        #         encoder_output = (batch_size,seq_length, hidden_units)
        #         coverage_vector = (batch_size,seq_length)

        # embedding look-up layer
        decoder_emb = self.embedding(decoder_input) # (batch_size, seq_length, hidden_units)

        # decoder_output = (batch_size,seq_length,hidden_units)
        # decoder_state = (batch_size,hidden_units)
        decoder_output , decoder_state = self.gru(decoder_emb,initial_state=decoder_state)

        # concatenate context vector and decoder state
        concat_vector = tf.concat([context_vector,decoder_state], axis=-1)
        # reshape to 1d array
        concat_vector = tf.reshape(concat_vector, (-1, concat_vector.shape[1]))
        # create vocabulary distribution
        p_vocab = tf.nn.log_softmax(self.W2(self.W1(concat_vector)))

        return p_vocab, decoder_state
