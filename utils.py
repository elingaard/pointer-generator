import tensorflow as tf
from tensorflow import keras

class Vocab(object):
    """Class for storing the mapping between words and their corresponding index in the vocabulary"""
    def __init__(self,tokenizer,max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # Total number of words in the Vocab
        self._word_to_id['<PAD>'] = self._count
        self._id_to_word[self._count] = '<PAD>'
        self._count += 1
        for _, word in tokenizer.index_word.items():
            self._word_to_id[word] = self._count
            self._id_to_word[self._count] = word
            self._count += 1
            if self._count >= max_size:
                break

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id['<UNK>']
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def decode_seq(self,seq):
        return " ".join([self._id_to_word[idx] for idx in seq])

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

def text2seq(text,tokenizer,vocab):
    """Convert a string or list of strings to a sequence of vocabulary ids

    Args:
    text(string or list of strings): text input
    tokenizer(object): tokenizer object
    vocab(object): vocabulary object

    Returns:
    seqs_padded(int tensor): sequence of vocabulary ids padded with start and end token ids
    """
    seqs = tokenizer.texts_to_sequences(text)
    seqs = [[vocab._word_to_id['<s>']]+seq+[vocab._word_to_id['<\s>']] for seq in seqs]
    max_len_seq = max([len(s) for s in seqs])
    seqs_padded = keras.preprocessing.sequence.pad_sequences(seqs, maxlen=max_len_seq, padding="post")
    return seqs_padded

def greedy_search(encoder_input,model,vocab,max_len_sum = 30):
    """Function which returns a summary by always picking the highest probability option conditioned on the previous word"""
    encoder_init_states = [tf.zeros((1, model.encoder.hidden_units)) for i in range(2)]
    encoder_output, encoder_states = model.encoder(encoder_input,encoder_init_states)
    decoder_state = encoder_states[0]

    decoder_input_t = tf.ones(1)*vocab._word_to_id['<s>']
    summary = [vocab._word_to_id['<s>']]
    coverage_vector = tf.zeros((1,encoder_input.shape[1]))
    while decoder_input_t[0].numpy()!=vocab._word_to_id['<\s>'] and len(summary)<max_len_sum:
        context_vector, attention_weights, coverage_vector = model.attention_model(decoder_state, encoder_output,coverage_vector)
        p_vocab, decoder_state = model.decoder(tf.expand_dims(decoder_input_t,1),decoder_state,encoder_output,context_vector)
        decoder_input_t = tf.argmax(p_vocab,axis=1)
        decoder_word_idx = int(decoder_input_t[0].numpy())
        summary.append(decoder_word_idx)
    return summary

def beam_search(encoder_input,model,vocab,beam_size=4,n_keep=4,max_len_sum=30):
    encoder_init_states = [tf.zeros((1, model.encoder.hidden_units)) for i in range(2)]
    encoder_output, encoder_states = model.encoder(encoder_input,encoder_init_states)
    decoder_state = encoder_states[0]

    coverage_vector = tf.zeros((1,encoder_input.shape[1]))
    candidates = [[0,[vocab._word_to_id['<s>']],[decoder_state,coverage_vector]]]
    not_terminated = True
    longest_sum = 0
    while not_terminated and longest_sum<max_len_sum:
        new_candidates = []
        for c_idx,cand in enumerate(candidates):
            if cand[1][-1]!=vocab._word_to_id['<\s>']:
                decoder_input_t = tf.ones(1)*cand[1][-1]
                decoder_state, coverage_vector = cand[2]
                context_vector, attention_weights, coverage_vector = model.attention_model(decoder_state, encoder_output,coverage_vector)
                p_vocab, decoder_state = model.decoder(tf.expand_dims(decoder_input_t,1),decoder_state,encoder_output,context_vector)
                values,indicies = tf.math.top_k(p_vocab,k=beam_size)
                for val,idx in zip(values.numpy()[0],indicies.numpy()[0]):
                    new_idx_list = cand[1] + [idx]
                    new_val = cand[0] + val
                    new_candidates.append([new_val,new_idx_list,[decoder_state, coverage_vector]])
            else:
                new_candidates.append(cand)
        candidates = sorted(new_candidates,key=lambda x:x[0]/len(x[1]),reverse=True)[:n_keep]
        not_terminated = sum([cand[1][-1]!=vocab._word_to_id['<\s>'] for cand in candidates])>0
        longest_sum = max([len(cand[1]) for cand in candidates])

    return candidates

def masked_nll_loss(p_vocab,target):
    """Calculate negative log-likelihood loss and use mask to ignore padding"""
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss = -p_vocab
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return loss

def coverage_loss(attention_weights,coverage_vector,target):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    coverage_vector = tf.expand_dims(coverage_vector,axis=2)
    ct_min = tf.reduce_min(tf.concat([attention_weights,coverage_vector],axis=2),axis=2)
    cov_loss = tf.reduce_sum(ct_min,axis=1)
    mask = tf.cast(mask, dtype=cov_loss.dtype)
    cov_loss *= mask
    return cov_loss
