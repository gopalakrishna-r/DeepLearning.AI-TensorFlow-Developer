import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, LSTMCell, Dense
from tensorflow.keras.models import Model
import tensorflow_addons as tfa


class NMTModel(Model):
    def __init__(self, vocab_size, embedding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_inputs = Input(shape=[None], dtype=np.int32)
        self.decoder_inputs = Input(shape=[None], dtype=np.int32)
        self.sequencer_lengths = Input(shape=[], dtype=np.int32)
        self.embeddings = Embedding(vocab_size, embedding_size)
        self.encoder = LSTM(512, return_state=True)

        self.decoder = tfa.seq2seq.basic_decoder.BasicDecoder(LSTMCell(512),
                                                              tfa.seq2seq.sampler.TrainingSampler(),
                                                              output_layer=Dense(vocab_size))

    def call(self, inputs, training=None, mask=None):
        encoder_in, decoder_in, seq_len = inputs
       # encoder_inputs = self.encoder_inputs(encoder_in)
        encoder_embeddings = self.embeddings(encoder_in)
        encoder_outputs, state_h, state_c = self.encoder(encoder_embeddings)
        encoder_state = [state_h, state_c]

        #decoder_inputs = self.decoder_inputs(decoder_in)
        decoder_embeddings = self.embeddings(decoder_in)

#        sequencer_len = self.sequencer_lengths(seq_len)
        final_outputs, final_state, final_sequence_lengths = \
            self.decoder(decoder_embeddings, initial_state=encoder_state,sequence_length=seq_len)
        return tf.nn.softmax(final_outputs.rnn_output)
