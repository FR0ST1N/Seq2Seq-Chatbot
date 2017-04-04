# Seq2Seq-Chatbot
Seq2Seq Chatbot Using Tensorflow.

## Dependencies

* Tensorflow
* Numpy
* Six

## Usage

To preapre the enc and dec file for your own data use the [pre_processing.py][1] inside the data dir.

1. Start training the model by changing the `mode = 'train'` in [bot.py][2].
2. Model will be saved in the checkpoint dir every few steps based on the value assigned to `steps_per_checkpoint`.
3. After training is finished set `mode = 'test'` and execute the script.

### Triggers

If you are training a closed domain chatbot then you can use the [trigger.py][3] to make the chatbot perform an action based on the decoded output.

* Place the triggers in the [dec][4] file and make sure the trigger symbol is in [vocab20000.dec][5].
* Update the [trigger.json][6].
* Update the conditional statement and the action to perform in [trigger.py][3].
* Train and run your model.

### Other Files

* [telegram.py][7] - Just assign the API key to the `bot` var to interact with your bot on telegram.
* [debug.py][8] - Prints everything.

## References

1. [A Neural Conversational Model][9]
2. [Tensorflow Sequence-to-Sequence Models][10]

[1]: https://github.com/FR0ST1N/Seq2Seq-Chatbot/blob/master/data/pre_processing.py
[2]: https://github.com/FR0ST1N/Seq2Seq-Chatbot/blob/master/bot.py
[3]: https://github.com/FR0ST1N/Seq2Seq-Chatbot/blob/master/trigger.py
[4]: https://github.com/FR0ST1N/Seq2Seq-Chatbot/blob/master/data/train.dec
[5]: https://github.com/FR0ST1N/Seq2Seq-Chatbot/blob/master/checkpoint/vocab20000.dec
[6]: https://github.com/FR0ST1N/Seq2Seq-Chatbot/blob/master/trigger.json
[7]: https://github.com/FR0ST1N/Seq2Seq-Chatbot/blob/master/telegram.py
[8]: https://github.com/FR0ST1N/Seq2Seq-Chatbot/blob/master/debug.py
[9]: https://arxiv.org/abs/1506.05869
[10]: https://www.tensorflow.org/tutorials/seq2seq
