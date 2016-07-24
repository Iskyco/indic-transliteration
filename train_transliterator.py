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
    """
    Parses file_name and returns en-hi transliteration pairs
    """
    en_hi_lines = open(file_name).read().strip().split('\n')
    en_hi_pairs = [ [x.strip().lower() for x in line.split('\t')] 
                                                for line in en_hi_lines]

    return en_hi_pairs

def get_char_vocab(script):
    """
    Returns char lookup list and dict for specified script.
    script can be one of 'devanagari' or 'latin'
    """
    if script=='devanagari':
        chars = _START_CHARS + list(map(chr, range(0x900, 0x97F)))
    else:
        chars = _START_CHARS + list(string.ascii_letters)

    chars_dict = {x:i for i,x in enumerate(chars)}
    return chars, chars_dict

def word_to_char_ids(word, char_dict, length=15):
    """
    Given a word and char_dict, returns list of ids padding to required length
    Characters not in char_dict are given id=unk_id
    """
    char_ids = [char_dict.get(x, UNK_ID) for x in word] + [EOS_ID]
    if len(char_ids) < length:
        char_ids += [PAD_ID]*(length-len(char_ids))
    
    return char_ids

def batch_generator(pairs, en_dict, hi_dict, batch_size, bucket):
    """
    Generates batches of data from pairs. bucket is pair of two integers 
    which will be used to pad the pairs
    """
    np.random.shuffle(pairs)
    
    for i in range(0, len(pairs)-batch_size, batch_size):
        X = [word_to_char_ids(word_pair[0], en_dict, bucket[0])
                for word_pair in pairs[i:i+batch_size]]
        
        Y = [word_to_char_ids(word_pair[1], hi_dict, bucket[1])
                for word_pair in pairs[i:i+batch_size]]
        
        X = np.array(X).T
        Y = np.array(Y).T
        
        yield X,Y

# TODO: Convert to class
def train_seq2seq():
    bucket = (16,16)
    batch_size = 64
    embedding_dim = 50
    memory_dim = 100
    num_layers = 2

    num_epochs = 100
    learning_rate = 0.05
    momentum = 0.9

    logdir = tempfile.mkdtemp()
    print("Logging at {}".format(logdir))

    # Get data
    en_hi_pairs = get_transliteration_pairs('data/Hindi - Word Transliteration Pairs 1.txt')
    en_hi_pairs = [x for x in en_hi_pairs if len(x[0])+1 < bucket[0] and len(x[1])+1 < bucket[1]]
    en_chars, en_dict = get_char_vocab('latin')
    hi_chars, hi_dict = get_char_vocab('devanagari')

    # Create model
    enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                        name="inp%i" % t)
                    for t in range(bucket[0])]

    labels = [tf.placeholder(tf.int32, shape=(None,),
                        name="labels%i" % t)
                    for t in range(bucket[1])]

    weights = [tf.ones_like(labels_t, dtype=tf.float32)
                    for labels_t in labels]

    # Decoder input: prepend some "GO" token and drop the final
    # token of the encoder input
    dec_inp = ([GO_ID*tf.ones_like(labels[0], dtype=np.int32, 
                        name="GO")] + labels[:-1])


    single_cell = tf.nn.rnn_cell.GRUCell(memory_dim)
    cell = single_cell#tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

    # Sequence to sequence model
    dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(
        enc_inp, dec_inp, cell, len(en_chars), len(hi_chars) ,embedding_dim)

    # loss and optimizer
    loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights)
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
            print("Epoch: ",i)
            for X, Y in batch_generator(en_hi_pairs, en_dict, hi_dict, batch_size, bucket):
                feed_dict = {enc_inp[t]: X[t] for t in range(bucket[0])}
                feed_dict.update({labels[t]: Y[t] for t in range(bucket[1])})
                
                _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)        
                summary_writer.add_summary(summary, step)
                
                # Sample the output every few epochs
                if step%100 == 0:
                    print("Step {}:".format(step), "Loss: {}".format(loss_t))
                    outs_soft = sess.run(dec_outputs, feed_dict)
                    outs = np.array([logits_t.argmax(axis=1) for logits_t in outs_soft])
                    for j in range(10):
                        print('\t', ''.join([en_chars[x] for x in X[:,j] if x>3]) , ' : ', 
                              ''.join([hi_chars[x] for x in outs[:,j] if x>3]))
                
                step = step + 1
                                    
            summary_writer.flush()


if __name__ == '__main__':
    train_seq2seq()