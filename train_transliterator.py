"""Trains seq2seq model in tensorflow."""
import string
import numpy as np
import tensorflow as tf
import tempfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_CHARS = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def get_transliteration_pairs(file_name):
    """Parse file_name and return en-hi transliteration pairs."""
    en_hi_lines = open(file_name).read().strip().split('\n')
    en_hi_pairs = [[x.strip().lower() for x in line.split('\t')]
                   for line in en_hi_lines]

    return en_hi_pairs


class Transliterator(object):
    """docstring for Transliterator."""

    def __init__(self, en_hi_pairs, bucket=(16, 16), embedding_dim=50,
                 memory_dim=100, num_layers=1):
        """Create tensorflow model."""
        # Get data and Remove pairs not fitting in buckets
        self.en_hi_pairs = en_hi_pairs
        self.en_hi_pairs = [x for x in self.en_hi_pairs if
                            len(x[0]) + 1 < bucket[0] and
                            len(x[1]) + 1 < bucket[1]]
        self.en_chars, self.en_dict = self.get_char_vocab('latin')
        self.hi_chars, self.hi_dict = self.get_char_vocab('devanagari')

        self.bucket = bucket
        self.embedding_dim = embedding_dim
        self.memory_dim = memory_dim
        self.num_layers = num_layers

        # Create_model creates tensorflow graphs
        self.create_model()

    def create_model(self):
        """Create tensorflow variables and graph."""
        self.enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                                       name="inp%i" % t)
                        for t in range(self.bucket[0])]

        self.labels = [tf.placeholder(tf.int32, shape=(None,),
                                      name="labels%i" % t)
                       for t in range(self.bucket[1])]

        self.weights = [tf.ones_like(labels_t, dtype=tf.float32)
                        for labels_t in self.labels]

        # Decoder input: prepend some "GO" token and drop the final
        # token of the encoder input
        self.dec_inp = ([GO_ID * tf.ones_like(self.labels[0], dtype=np.int32,
                                              name="GO")] + self.labels[:-1])

        single_cell = tf.nn.rnn_cell.LSTMCell(self.memory_dim)
        if self.num_layers > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell(
                [single_cell] * self.num_layers)
        else:
            self.cell = single_cell

        # Sequence to sequence model
        self.dec_outputs, self.dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(
            self.enc_inp, self.dec_inp, self.cell, len(self.en_chars),
            len(self.hi_chars), self.embedding_dim)

    @staticmethod
    def word_to_char_ids(word, char_dict, length=15):
        """Given a word and char_dict, returns list of padded ids.
        Characters not in char_dict are given id=unk_id
        """
        char_ids = [char_dict.get(x, UNK_ID) for x in word] + [EOS_ID]
        if len(char_ids) < length:
            char_ids += [PAD_ID] * (length - len(char_ids))

        return char_ids

    @staticmethod
    def get_char_vocab(script):
        """
        Returns char lookup list and dict for specified script.
        script can be one of 'devanagari' or 'latin'
        """
        if script == 'devanagari':
            chars = _START_CHARS + list(map(chr, range(0x900, 0x97F)))
        else:
            chars = _START_CHARS + list(string.ascii_letters)

        chars_dict = {x: i for i, x in enumerate(chars)}
        return chars, chars_dict

    def data_generator(self, pairs, batch_size, bucket):
        """
        Generates batches of data from pairs. bucket is pair of two integers
        which will be used to pad the pairs
        """
        np.random.shuffle(pairs)

        for i in range(0, len(pairs) - batch_size, batch_size):
            X = [self.word_to_char_ids(word_pair[0], self.en_dict, bucket[0])
                 for word_pair in pairs[i:i + batch_size]]
            [x.reverse() for x in X]

            Y = [self.word_to_char_ids(word_pair[1], self.hi_dict, bucket[1])
                 for word_pair in pairs[i:i + batch_size]]

            X = np.array(X).T
            Y = np.array(Y).T

            yield X, Y

    def train(self, learning_rate=0.05, momentum=0.9, batch_size=64, 
              num_epochs=100):
        """
        Trains model
        """
        # Create tempdir for logging
        logdir = tempfile.mkdtemp()
        print("Logging at {}".format(logdir))

        # loss and optimizer
        loss = tf.nn.seq2seq.sequence_loss(
            self.dec_outputs, self.labels, self.weights)
        tf.scalar_summary("loss", loss)
        summary_op = tf.merge_all_summaries()
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        train_op = optimizer.minimize(loss)

        # Run!
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            summary_writer = tf.train.SummaryWriter(logdir, sess.graph)

            step = 0
            for i in range(num_epochs):
                print("Epoch: ", i)
                for X, Y in self.data_generator(self.en_hi_pairs, batch_size, self.bucket):
                    feed_dict = {self.enc_inp[t]: X[t]
                                 for t in range(self.bucket[0])}
                    feed_dict.update({self.labels[t]: Y[t]
                                      for t in range(self.bucket[1])})

                    _, loss_t, summary = sess.run(
                        [train_op, loss, summary_op], feed_dict)
                    summary_writer.add_summary(summary, step)

                    # Sample the output every few epochs
                    if step % 100 == 0:
                        print("Step {}:".format(step),
                              "Loss: {}".format(loss_t))
                        outs_soft = sess.run(self.dec_outputs, feed_dict)
                        outs = np.array([logits_t.argmax(axis=1)
                                         for logits_t in outs_soft])
                        for j in range(10):
                            print('\t', ''.join(reversed([self.en_chars[x] for x in X[:, j] if x > 3])), ' : ',
                                  ''.join([self.hi_chars[x] for x in outs[:, j] if x > 3]))

                    step = step + 1

                summary_writer.flush()


if __name__ == '__main__':
    trans = Transliterator(get_transliteration_pairs(
        'data/Hindi - Word Transliteration Pairs 1.txt'))
    trans.train()
