"""

                offense.py

Put your code for generating adversarial attacks here.

The throwdown code will look for one function here: generate_adversarial_example, which
should input the location of a saved model, an MNIST image, and the correct labe, and
output an adversarial example.

"""

import numpy as np
import tensorflow as tf




def fgsm(logdir, x_adv, label, epsilon=0.01, num_iterations=1, targeted=False):
    """
    Use the Fast Gradient Sign Method to build adversarial examples
    
    :logdir: directory where the saved TF model is stored. Expects to find "x", "y_", 
            and "logits" tensors
    :x_adv: (N, 784) array of data points to build examples from
    :label: (N,) array of corresponding correct labels (if targeted=False) or
            target labels (if targeted=True)
    :epsilon: step size for shifting the examples
    :num_iterations: how many times to update the examples
    :targeted: whether to use targeted fast gradient
    
    Returns
    :x_adv: (N, 784) array of adversarial examples
    :probs: (N, C) array of softmax probabilities
    """
    x_adv = x_adv.copy()
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        # Load the saved network
        tf.saved_model.loader.load(sess, ["tag"], logdir)
        # Pull out tensors for the inputs
        x = graph.get_tensor_by_name("x:0")# the ":0" indexes the operation's output tensor
        y_ = graph.get_tensor_by_name("y_:0")
        # Pull out tensors for the logits
        logits = graph.get_tensor_by_name("logits:0")
        # compute outputs 
        y = tf.nn.softmax(logits)
    
        # build a loss function
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                   labels=tf.one_hot(y_, 10))
        # find derivative of loss with respect to x
        dy_dx = tf.gradients(loss, x)[0]
        
        if targeted:
            epsilon *= -1
    
        # for each iteration
        for i in range(num_iterations):
            # compute the derivative with respect to the loss
            d = sess.run(dy_dx, {x:x_adv, y_:label})
            # save the new image by shifting the old one by epsilon * sign(gradient)
            x_adv += epsilon * np.sign(d)
        # compute softmax outputs for the examples
        probs = sess.run(y, {x:x_adv})
    return x_adv, probs


def generate_adversarial_example(logdir, original, correct_label):
    """
    Corrupt an MNIST example for an adversarial attack
    
    :logdir: string; directory containing saved model
    :original: a length-784 numpy array with the original MNIST example
    :correct_label: integer; the correct label
    
    Returns a length-784 numpy array with an adversarial example
    """
    # go nuts here
    return fgsm(logdir, original.reshape(1,-1), np.array([correct_label]),
               epsilon=0.01, num_iterations=5)[0]


def generate_adversarial_example2(logdir, original, correct_label):
    """
    Corrupt an MNIST example for an adversarial attack
    
    :logdir: string; directory containing saved model
    :original: a length-784 numpy array with the original MNIST example
    :correct_label: integer; the correct label
    
    Returns a length-784 numpy array with an adversarial example
    """
    # go nuts here
    return fgsm(logdir, original.reshape(1,-1), np.array([correct_label]),
               epsilon=0.01, num_iterations=10)[0]


