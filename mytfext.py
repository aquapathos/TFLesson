import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import seaborn as sns
import tensorflow as tf

def plotchr(image,label,col,i):
    sns.set_context("talk")
    plt.subplot(1,col,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title("%d" % np.argmax(label))
    plt.imshow(image,cmap=plt.cm.gray_r)
