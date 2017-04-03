from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import json
from pprint import pprint

import numpy as np
from six.moves import xrange  
import tensorflow as tf

import data_utils
import seq2seq_model

mode = 'test'
train_enc = 'data/train.enc'
train_dec = 'data/train.dec'
working_directory = 'checkpoint/'
enc_vocab_size = 20000
dec_vocab_size = 20000
num_layers = 3
layer_size = 256
max_train_data_size = 0
batch_size = 64
steps_per_checkpoint = 1000
learning_rate = 0.5
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0
lines = []

try:
    reload
except NameError:
    pass
else:
    reload(sys).setdefaultencoding('utf-8')

    



_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
 
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):

 #Create model and initialize or load parameters
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


def train():
  # prepare dataset
  print("Starting to train from " + working_directory)
  enc_train, dec_train, _, _ = data_utils.prepare_custom_data(working_directory,train_enc,train_dec,enc_vocab_size,dec_vocab_size)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
  config = tf.ConfigProto(gpu_options=gpu_options)
  config.gpu_options.allocator_type = 'BFC'

  with tf.Session(config=config) as sess:
    print("Creating model with %d layers and %d cells." % (num_layers, layer_size))
    model = create_model(sess, False)
    train_set = read_data(enc_train, dec_train, max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))


    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    count = 0
    while True:
      count += 1
      print('Step: ' + str(count))
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / steps_per_checkpoint
      loss += step_loss / steps_per_checkpoint
      current_step += 1

      if current_step % steps_per_checkpoint == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("Saved model at step %d with perplexity %.2f "
                % (model.global_step.eval(),
                         perplexity))
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        checkpoint_path = os.path.join(working_directory, "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        sys.stdout.flush()


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
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
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
      if sentence[:-1] in lines:
        temp_output = " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
        trigger_check = trigger_activator(temp_output)
        if trigger_check == True:
            print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs[:-1]]))
        else:
            print(temp_output)
      else:
          print('i dont understand you')      
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

#Check if there is a trigger in the decoded sentence
def trigger_activator(decoded_output):
    if "_trigger" in decoded_output: 
        trigger_value = decoded_output.split(' ')[-1:]
        trigger_words = decoded_output.split(' ')
        trigger_value = str(trigger_value).replace("'","").replace("]","").replace("[","")
        for x in range(0, len(data['triggers'])):
          if data['triggers'][x]['id'] == str(trigger_value):
            json_words = data['triggers'][x]['keywords']
            if json_words[0] in decoded_output:
              if json_words[1] in decoded_output:
                if str(trigger_value) == "_trigger1":
                  print('Your action here!!')
                  return True


if __name__ == '__main__':

    print('Starting the script...')

    if mode == 'train':
        train()
    else:
        #Load the trigger file
        with open('tirgger.json') as data_file:    
          data = json.load(data_file)
        #Reply only to the trained inputs
        '''input_file = open('data/train.enc', 'rw')
        for line in input_file:
          lines.append(line.lower().replace('\n','').replace('\r',''))'''
        decode()