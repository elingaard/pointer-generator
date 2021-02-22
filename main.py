import os
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
from model import Encoder, BahdanauAttention, Decoder
from utils import Vocab, text2seq, masked_nll_loss, coverage_loss

class PointerGenerator:
    def __init__(self,args):
        self.datapath = args.DATAPATH
        self.batch_size = args.batch_size
        self.n_epochs = args.epochs
        self.vocab_size = args.vdim #
        self.embedding_dim = args.edim
        self.hidden_dim = args.hdim
        self.encoder = Encoder(self.vocab_size+2, self.embedding_dim, self.hidden_dim, embedding_matrix=None) #+2 on vocab size due to start and end token
        self.attention_model = BahdanauAttention(self.hidden_dim,is_coverage=True)
        self.decoder = Decoder(self.vocab_size+2, self.embedding_dim, self.hidden_dim, embedding_matrix=None)
        self.optimizer = keras.optimizers.Adam()

    #@tf.function
    def train_step(self,encoder_input, decoder_target):
        """Function which performs one training step (batch)"""
        loss = tf.zeros(self.batch_size)
        lambda_cov = 1
        with tf.GradientTape() as tape:
            # run body_sequence input through encoder
            encoder_init_states = [tf.zeros((self.batch_size, self.hidden_dim)) for i in range(2)]
            encoder_output, encoder_states = self.encoder(encoder_input,encoder_init_states)
            # initialize decoder with encoder forward state
            decoder_state = encoder_states[0] # !!!interpolate between forward and backward instead!!!
            coverage_vector = tf.zeros((self.batch_size,encoder_input.shape[1]))
            # loop over each word in target sequence
            for t in range(decoder_target.shape[1]-1):
                # run decoder input through decoder and generate vocabulary distribution
                decoder_input_t = decoder_target[:,t]
                decoder_target_t = decoder_target[:,t+1]
                # get attention scores
                context_vector, attention_weights, coverage_vector = self.attention_model(decoder_state, encoder_output,coverage_vector)
                # get vocabulary distribution for each batch at time t
                p_vocab,decoder_state = self.decoder(tf.expand_dims(decoder_input_t,1),decoder_state,encoder_output,context_vector)
                # for each batch get the probability of the target word at time t+1
                p_vocab_list = []
                for i in range(len(decoder_target_t)):
                    p_vocab_list.append(p_vocab[i,decoder_target_t[i]])
                p_vocab_target = tf.stack(p_vocab_list)
                # calculate the loss at each time step t and add to current loss
                loss += masked_nll_loss(p_vocab_target,decoder_target_t) + lambda_cov*coverage_loss(attention_weights,coverage_vector,decoder_target_t)

            # get the non-padded length of each sequence in the batch
            seq_len_mask = tf.cast(tf.math.logical_not(tf.math.equal(decoder_target, 0)),tf.float32)
            batch_seq_len = tf.reduce_sum(seq_len_mask,axis=1)

            # get batch loss by dividing the loss of each batch by the target sequence length and mean
            batch_loss = tf.reduce_mean(loss/batch_seq_len)

        # update trainable variables
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


    def train(self):
        # load text data
        with open(os.path.join(self.datapath,'tok.train.abstract.txt'),'rb') as f:
            body_data_train = f.read().decode("utf-8").split('\n')
        with open(os.path.join(self.datapath,'tok.train.title.txt'),'rb') as f:
            target_data_train = f.read().decode("utf-8").split('\n')
        with open(os.path.join(self.datapath,'tok.valid.abstract.txt'),'rb') as f:
            body_data_valid = f.read().decode("utf-8").split('\n')
        with open(os.path.join(self.datapath,'tok.valid.title.txt'),'rb') as f:
            target_data_valid = f.read().decode("utf-8").split('\n')

        # define vocabulary and tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size, oov_token='<UNK>')
        tokenizer.fit_on_texts(body_data_train)
        tokenizer.index_word[self.vocab_size] = '<s>' # add sentence start token
        tokenizer.index_word[self.vocab_size+1] = '<\s>' # add sentence end token
        vocab = Vocab(tokenizer,self.vocab_size+2)

        # create text sequences
        body_seqs_train = text2seq(body_data_train,tokenizer,vocab)
        target_seqs_train = text2seq(target_data_train,tokenizer,vocab)
        body_seqs_valid = text2seq(body_data_valid,tokenizer,vocab)
        target_seqs_valid = text2seq(target_data_valid,tokenizer,vocab)

        # create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((body_seqs_train,target_seqs_train))
        train_dataset = train_dataset.shuffle(len(body_seqs_train)).batch(self.batch_size, drop_remainder=True)
        valid_dataset = tf.data.Dataset.from_tensor_slices((body_seqs_valid,target_seqs_valid))
        valid_dataset = valid_dataset.shuffle(len(body_seqs_valid)).batch(self.batch_size, drop_remainder=True)

        # run one batch through model to initialize parameters
        encoder_input, decoder_target = next(iter(train_dataset))
        encoder_init_states = [tf.zeros((self.batch_size, self.hidden_dim)) for i in range(2)]
        encoder_output, encoder_states = self.encoder(encoder_input,encoder_init_states)
        decoder_state = encoder_states[0]
        coverage_vector = tf.zeros((self.batch_size,encoder_input.shape[1]))
        decoder_input_t = decoder_target[:,0]
        context_vector, attention_weights, coverage_vector = self.attention_model(decoder_state, encoder_output,coverage_vector)
        p_vocab,decoder_state = self.decoder(tf.expand_dims(decoder_input_t,1),decoder_state,encoder_output,context_vector)

        # training loop
        epoch_loss = tf.keras.metrics.Mean()
        for epoch in range(self.n_epochs):
            epoch_loss.reset_states()

            with tqdm(total=len(body_seqs_train) // self.batch_size) as batch_progress:
                for batch, (encoder_input, decoder_target) in enumerate(train_dataset):
                    batch_loss = self.train_step(encoder_input, decoder_target)
                    epoch_loss(batch_loss)

                    if (batch % 10) == 0:
                        batch_progress.set_description(f'Epoch {epoch + 1}')
                        batch_progress.set_postfix(Batch=batch, Loss=batch_loss.numpy())
                        batch_progress.update()

            #self.eval()

    def eval(self):
        # greedy_search, beam_search
        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATAPATH',default="title-gen-5m-tok",type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--edim', default=256, type=int)
    parser.add_argument('--hdim', default=128, type=int)
    parser.add_argument('--vdim', default=20000, type=int)
    args = parser.parse_args()

    pointgen = PointerGenerator(args)
    pointgen.train()





















