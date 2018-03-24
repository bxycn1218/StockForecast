import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# get train data
def get_train_data(data, input_size, output_size, batch_size, time_step, train_begin, train_end):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # normalize
    train_x, train_y = [], []  # train set
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :input_size]
        y = normalized_train_data[i:i + time_step, input_size, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# get test data
def get_test_data(data, input_size, output_size, time_step, test_begin):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # normalize
    size = (len(normalized_test_data) + time_step - 1) // time_step  # sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :7]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :input_size]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, input_size]).tolist())
    return mean, std, test_x, test_y


# define lstm network
def lstm(data, input_size, output_size, re_use=False, rnn_unit=10):
    batch_size = tf.shape(data)[0]
    time_step = tf.shape(data)[1]
    with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
        weights = {
            'in': tf.get_variable(name='weights-in', shape=[input_size, rnn_unit],
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0)),
            'out': tf.get_variable(name='weights-out', shape=[rnn_unit, 1],
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        }
        biases = {
            'in': tf.get_variable(name='bias-in', shape=[rnn_unit, ], initializer=tf.constant_initializer(0.1)),
            'out': tf.get_variable(name='bias-out', shape=[1, ], initializer=tf.constant_initializer(0.1))
        }
        w_in = weights['in']
        b_in = biases['in']
        input = tf.reshape(data, [-1, input_size])  # reshape into 2d
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # reshape into 3d
        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        output = tf.reshape(output_rnn, [-1, rnn_unit])
        w_out = weights['out']
        b_out = biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states


# training function
def train_lstm(data, input_size=7, output_size=1, learning_rate=0.001, batch_size=60, time_step=20, train_begin=0, train_end=5800):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(data, input_size, output_size, batch_size, time_step, train_begin, train_end)
    pred, _ = lstm(X, input_size, output_size)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):  # iterate times
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
            print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ", saver.save(sess, 'model_lstm/modle.ckpt'))    # save model
        print("The train has finished")


# prediction function
def prediction(data, input_size=7, output_size=1, time_step=20, test_begin=5800):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(data, input_size, output_size, time_step, test_begin)
    pred, _ = lstm(X, input_size, output_size)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # restore variable
        module_file = tf.train.latest_checkpoint('model_lstm')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[7] + mean[7]
        test_predict = np.array(test_predict) * std[7] + mean[7]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # acc
        mae = np.average(np.abs(test_predict - test_y[:len(test_predict)]))  # mae
        return test_predict, test_y, acc, mae
