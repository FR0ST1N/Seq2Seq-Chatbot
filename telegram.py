from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  
import tensorflow as tf
import telebot

import data_utils
import seq2seq_model

train_enc = 'data/train.enc'
train_dec = 'data/train.dec'
working_directory = 'checkpoint/'
enc_vocab_size = 20000
dec_vocab_size = 20000
num_layers = 3
layer_size = 256
max_train_data_size = 0
batch_size = 64
steps_per_checkpoint = 5000
learning_rate = 0.5
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0

try:
    reload
except NameError:
    pass
else:
    reload(sys).setdefaultencoding('utf-8')

bot = telebot.TeleBot("")  #Your API Key Here



_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def create_model(session, forward_only):

  model = seq2seq_model.Seq2SeqModel( enc_vocab_size, dec_vocab_size, _buckets, layer_size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(working_directory)
  checkpoint_suffix = ""
  if tf.__version__ > "0.12":
      checkpoint_suffix = ".index"
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
    print("Model detected at %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Creating a new model.")
    session.run(tf.global_variables_initializer())  
  return model

def decode():
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
  config = tf.ConfigProto(gpu_options=gpu_options)

  with tf.Session(config=config) as sess:
    model = create_model(sess, True)
    model.batch_size = 1 
    enc_vocab_path = os.path.join(working_directory,"vocab%d.enc" % enc_vocab_size)
    dec_vocab_path = os.path.join(working_directory,"vocab%d.dec" % dec_vocab_size)

    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)
    print('Start chatting...')
    @bot.message_handler(func=lambda sentence: True)
    def reply_all(message):
        sentence = (message.text).lower()
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
        bucket_id = min([b for b in xrange(len(_buckets))
                        if _buckets[b][0] > len(token_ids)])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                        target_weights, bucket_id, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        message_text = " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
        bot.reply_to(message, message_text)
    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as ex:
            print(str(ex))
            bot.stop_polling()
            time.sleep(5)
            bot.polling()


if __name__ == '__main__':
    decode()


