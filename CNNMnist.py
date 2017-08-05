
# coding: utf-8

# # 多層CNNによるMNISTの手書き数字認識
# http://tensorflow.classcat.com/2016/03/10/tensorflow-cc-deep-mnist-for-experts/

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import seaborn as sns
import tensorflow as tf
import mytfext
get_ipython().magic('matplotlib inline')

# ※ tensorflow にはMNISTのサンプルデータをダウンロードするためのモジュールが備わっている
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# # 重みの初期化関数
def weight_variable(shape, name=""):
    initial = tf.truncated_normal(shape, stddev=0.1)
    if name == "":
        return tf.Variable(initial)
    else:
        return tf.Variable(initial,name=name)

def bias_variable(shape, name=""):
    initial = tf.constant(0.1, shape=shape)
    if name == "":
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=mane)


# # 畳み込みとプーリングの関数
def conv2d(x, W, name=""):
    if name =="":
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    else:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool(x,ks,ss, name=""):
    if name == "":
        return tf.nn.max_pool(x, ksize=[1, ks, ks, 1],
                        strides=[1, ss, ss, 1], padding='SAME')
    else:
        return tf.nn.max_pool(x, ksize=[1, ks, ks, 1],
                        strides=[1, ss, ss, 1], padding='SAME', name=name)


# # モデルの定義
# モデルの作成
np.random.seed(19601228)
tf.set_random_seed(19601228)

# num_filters1 = 4
# num_filters2 = 4
# num_hidden_units = 196

class NN:
    def __init__(self, learning_rate=0.001,num_filters1 = 32,num_filters2 = 64,num_hidden_units = 1024):
        with tf.Graph().as_default():
            self.model(learning_rate)
            self.session()

    def model(self, learning_rate,num_filters1 = 32,num_filters2 = 64,num_hidden_units = 1024):
        with tf.name_scope("CN1"):
            ideal = tf.placeholder(tf.float32, [None, 10], name="ideal")
            img = tf.placeholder(tf.float32, [None, 784], name="input_data")

            x_image = tf.reshape(img, [-1,28,28,1])
            convM1 = weight_variable([5,5,1,num_filters1], name="convM1")
            b_conv1 = bias_variable([num_filters1])
            h_conv1 = tf.nn.relu(conv2d(x_image, convM1)+b_conv1)
            h_pool1 = max_pool(h_conv1, ks=2,ss=2, name = "pool1")

        with tf.name_scope("CN2"):
            convM2 = weight_variable([5,5,num_filters1,num_filters2], name="convM2")
            b_conv2 = bias_variable([num_filters2])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, convM2)+b_conv2)
            h_pool2 = max_pool(h_conv2,ks=2,ss=2, name="pool2")

        with tf.name_scope("hidden"):
            nflathpool2 = 7*7*num_filters2
            h_pool2Flat = tf.reshape(h_pool2,[-1,nflathpool2])
            W_hidden = weight_variable([nflathpool2, num_hidden_units], name="W_hidden")
            b_hidden = bias_variable([num_hidden_units])
            h_hidden = tf.nn.relu(tf.matmul(h_pool2Flat,W_hidden)+b_hidden, name="h_hidden")

            keep_prob = tf.placeholder(tf.float32) # ドロップアウト時のキープ率のプレースフォルダ
            h_keep = tf.nn.dropout(h_hidden,keep_prob, name="h_keep")

        with tf.name_scope("output"):
            W_out = weight_variable([num_hidden_units, 10], name="W_out")
            b_out = bias_variable([10])
            # out = tf.nn.softmax(tf.matmul(h_keep,W_out)+b_out, name="softmax-output")
            out = tf.nn.softmax(tf.matmul(h_hidden,W_out)+b_out, name="softmax-output")

        with tf.name_scope("Optimizer") as scope:
            loss = -tf.reduce_sum(ideal*tf.log(out), name="loss")
            # train_step = tf.train.GradientDescentOptimizer(0.00002).minimize(loss)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        with tf.name_scope("Evaluator") as scope:
            correct_prediction = tf.equal(tf.argmax(out,1),tf.argmax(ideal,1), name="correct")
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        self.ideal = ideal
        self.img = img
        self.keep_prob = keep_prob

        self.out = out
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy
        self.convM1, self.convM2, self.W_hidden, self.W_out = convM1,convM2, W_hidden, W_out
        self.h_pool1,self.h_pool2 = h_pool1, h_pool2

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("Matrix1", convM1)
        tf.summary.histogram("Matrix2", convM2)
        tf.summary.histogram("W_hidden", W_hidden)
        tf.summary.histogram("W_pool1", h_pool1)
        tf.summary.histogram("W_pool2t", h_pool2)
        tf.summary.histogram("out", out)

    def session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # sess.run(tf.local_variables_initializer())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log9", sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer
        self.saver = saver


# # セッションの開始

nn = NN(learning_rate=0.001)
times = 0
for _ in range(2000):
    times += 1
    batch_xs,batch_ts = mnist.train.next_batch(30)
    nn.sess.run(nn.train_step, feed_dict={nn.keep_prob:0.5,nn.img:batch_xs, nn.ideal:batch_ts})
    if times % 100 == 0:
        summary,loss_val, acc_val = nn.sess.run([nn.summary,nn.loss,nn.accuracy],
                feed_dict={nn.keep_prob:1.0,nn.img:mnist.test.images, nn.ideal:mnist.test.labels})
        print("Step: {0:d}, Loss: {1:f}, Accuracy: {2:f}".format(times, loss_val, acc_val))
        nn.writer.add_summary(summary,times)
        # nn.saver.save(nn.sess, '.\mysession', global_step = times)


# # 98%!
# 畳み込み層を１段目２段目とも４、隠れ層196でも98％
#
# 普通の隠れ層１のNNや畳み込み層１隠れ層なしのNNでも98％は到達するのだから当然なのかもしれない。

# # 学習結果を用いて文字認識してみる

plt.figure(figsize=(8,15))
def plotchr(image,label1,label2, i):
    sns.set_context("talk")
    plt.subplot(10,6,i)
    plt.xticks([])
    plt.yticks([])
    plt.title("%d / %d" % (np.argmax(label1),np.argmax(label2)))
    plt.imshow(image,cmap=plt.cm.gray_r)

images, labels = mnist.test.images, mnist.test.labels
p_val = nn.sess.run(nn.out, feed_dict={nn.img:images, nn.ideal:labels})

# 0～9 と判定されたものについて、それぞれ正の例を３つ、負の例を３つ表示する
picnumber = 0
for i in range(10):
    c = 0
    for (image, label, pred) in zip(images, labels, p_val):
        prediction,actual = np.argmax(pred), np.argmax(label)
        if prediction != i:  #
            continue
        if(c<3 and i == actual) or (c>=3 and i != actual):
            picnumber += 1
            plotchr(image.reshape((28,28)),pred,label,picnumber)
            c += 1
            if c > 5:
                break

num_filters=16

def showFilterEffect(num_filters, start, filters,imgs):
    fig = plt.figure(figsize=(10,num_filters+1))
    for i in range(num_filters):
        subplot = fig.add_subplot(num_filters+1, 10, 10*(i+1)+1)
        sns.heatmap(filters[:,:,0,i],yticklabels=[],xticklabels=[],
                cmap=plt.cm.bwr,cbar=False,annot=False,square=True)

    for i in range(9):
        subplot = fig.add_subplot(num_filters+1, 10, i+2)
        oimg = mnist.test.images[start+i].reshape((28,28))
        title = '{:d}'.format(np.argmax(mnist.test.labels[start+i]))
        subplot.set_title(title)
        sns.heatmap(oimg,yticklabels=[],xticklabels=[],
                cmap=plt.cm.binary,cbar=False,annot=False,square=True)

        for f in range(num_filters):
            subplot = fig.add_subplot(num_filters+1, 10, 10*(f+1)+i+2)
            sns.heatmap(imgs[i][:,:,f],yticklabels=[],xticklabels=[],
                cmap=plt.cm.binary,cbar=False,annot=False,square=True)

filter_vals, conv_vals, pool_vals = nn.sess.run(
    [nn.W_conv, nn.h_conv, nn.h_pool], feed_dict={nn.img:mnist.test.images[0:10]})
showFilterEffect(16,0,filter_vals,conv_vals)

showFilterEffect(16,0,filter_vals,pool_vals)

fig = plt.figure(figsize=(12,10))
c=0
for i in range(len(mnist.test.images)):
    image = mnist.test.images[i:i+1]
    label = mnist.test.labels[i]
    p_val = nn.sess.run(nn.out, feed_dict={nn.img:image})
    pred = p_val[0]
    prediction, actual = np.argmax(pred), np.argmax(label)
    if prediction == actual:
        continue
    subplot = fig.add_subplot(5,4,c*2+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('{} / {}'.format(prediction, actual))
    subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1,
                   cmap=plt.cm.binary, interpolation="nearest")
    subplot = fig.add_subplot(5,4,c*2+2)
    subplot.set_xticks(range(10))
    subplot.set_xlim(-0.5,9.5)
    subplot.set_ylim(0,1)
    subplot.bar(range(10), pred, align='center')
    c += 1
    if c == 10:
        break
