import tensorflow as tf
import numpy as np


# get train data
def get_train_data(data, input_size, output_size, batch_size, train_begin, train_end):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # normalize
    train_x, train_y = [], []  # train set
    for i in range(len(normalized_train_data) - 1):
        if i % batch_size == 0:
            batch_index.append(i)
        train_x.append(normalized_train_data[i,:input_size])
        train_y.append(normalized_train_data[i+1,input_size:input_size+output_size])
    batch_index.append((len(normalized_train_data) - 1))
    return batch_index, train_x, train_y


# get test data
def get_test_data(data, input_size, output_size, test_begin):
    data_test = data[test_begin-1:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # normalize
    test_x = normalized_test_data[:-1, :input_size]
    test_y = normalized_test_data[1:, input_size:input_size+output_size]
    return mean, std, test_x, test_y


# add neural layer
def add_layer(layer_name, re_use, input_data, in_size, out_size, activity_function=None):
    # reuse = auto, can be set as re_use
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name='weights', shape=[in_size, out_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        bias = tf.get_variable(name='bias', shape=[1, out_size], initializer=tf.constant_initializer(0.1))
        weights_plus_b = tf.matmul(input_data, weights) + bias
        if activity_function is None:
            ans = weights_plus_b
        else:
            ans = activity_function(weights_plus_b)
        return ans


def bp(data, input_size, output_size, re_use=False, unit=10):
    l1 = add_layer('l1', re_use, data, input_size, unit, activity_function=tf.nn.relu)  # activity function: relu
    l2 = add_layer('l2', re_use, l1, unit, output_size, activity_function=None)
    return l2


# training function
def train_bp(data, input_size=7, output_size=1, learning_rate=0.001, batch_size=60, train_begin=0, train_end=5800):
    batch_index, train_x, train_y = get_train_data(data, input_size, output_size, batch_size, train_begin, train_end)
    X = tf.placeholder(tf.float32, shape=[None, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, output_size])
    loss = tf.reduce_mean(tf.square((Y - bp(X, input_size, output_size))))  # loss function
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # optimize function
    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op,loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                         Y: train_y[batch_index[step]:batch_index[step + 1]]})
                print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ", saver.save(sess, 'model_bp/model.ckpt'))    # save model
        print("The train has finished")


# prediction function
def prediction(data, input_size=7, output_size=1, test_begin=5800):
    X = tf.placeholder(tf.float32, shape=[None, input_size])
    pred = bp(X, input_size, output_size, re_use=True)
    mean, std, test_x, test_y = get_test_data(data, input_size, output_size, test_begin)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # Restore variables from disk.
        module_file = tf.train.latest_checkpoint('model_bp')
        saver.restore(sess, module_file)
        test_predict = []
        for i in range(len(test_x)):
            predict = sess.run(pred, feed_dict={X: [test_x[i]]})
            test_predict.extend(predict.reshape((-1)))
        test_predict = np.array(test_predict) * std[7] + mean[7]
        test_y = np.array(test_y.flatten()) * std[7] + mean[7]
        acc = np.average(np.abs(test_predict - test_y)/test_y)  # ACC
        mae = np.average(np.abs(test_predict - test_y))  # mae
    return test_predict, test_y, acc, mae
