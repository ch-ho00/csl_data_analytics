from string import punctuation
from collections import Counter
import pandas  as pd
import numpy as np
import re
import tensorflow as tf

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def getSentenceMatrix(sentences):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    sentences = [cleanSentences(sent) for sent in sentences]
    split = " ".join(sentences).split()
    err = 0
    for indexCounter,word in enumerate(split): 
        if indexCounter < 250:
            try:
                sentenceMatrix[0,indexCounter] = wordsList.index(word)
            except ValueError:
                err += 1
                sentenceMatrix[0,indexCounter] = 399999 #Vector for unknown words
        else:
            break
    return sentenceMatrix, err

numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

wordsList = np.load('../LSTM-Sentiment-Analysis/wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('../LSTM-Sentiment-Analysis/wordVectors.npy')


strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('../LSTM-Sentiment-Analysis/models'))
# ________________________________________________________

review = pd.read_csv('./Data/reviews.csv', nrows=2000)
review['comments'] = review['comments'].apply(lambda x:[sent for sent in x.lower().replace(',','').replace('\n','').replace('\r','').split('.') if len(sent) > 3])
# all_text2  = [" ".join(l) for l in review2['comments'].values.tolist()]
# words = [l.split() for l in all_text2]
# count_words = Counter(words)
# total_words = len(words)
# sorted_words = count_words.most_common(total_words)
# print(count_words)
# vocab_to_int = {w:i for i, (w,c) in enumerate(sorted_words)}
# vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}


for i,text in review['comments'].iteritems():
    inputMatrix,err = getSentenceMatrix(text)
    predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
    # predictedSentiment[0] represents output score for positive sentiment
    # predictedSentiment[1] represents output score for negative sentiment

    if predictedSentiment[1] - predictedSentiment[0]  > 5 and  predictedSentiment[1] > 5:
        print(text)
        print ("Negative Sentiment",predictedSentiment[0],predictedSentiment[1])
        print(err,"____________________________")