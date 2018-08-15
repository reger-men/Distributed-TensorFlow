import tensorflow as tf
import numpy as np

# Get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Some numbers
batch_size = 1
display_step = 1
scale_factor = 100
o_input_size = 280
num_input = o_input_size*o_input_size*scale_factor*scale_factor
num_classes = 10

# Set up tf session and initialize variables.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
config.gpu_options.allocator_type = 'BFC'

def conv_layer(inputs, channels_in, channels_out, strides=1):

        # Create variables
        w=tf.Variable(tf.random_normal([1, 1, channels_in, channels_out]))
        b=tf.Variable(tf.random_normal([channels_out]))

        # We can double check the device that this variable was placed on
        print(w.device) 
        print(b.device)

        # Define Ops
        x = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)

        # Non-linear activation
        return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def CNN(x):
    x = tf.reshape(x, shape=[-1, o_input_size*scale_factor, o_input_size*scale_factor, 1])

    with tf.device('/gpu:1'):
        # Convolution Layer
        conv1=conv_layer(x, 1, 2, strides=1)
    return conv1

with tf.device('/cpu:0'):
    # Construct model
    X = tf.placeholder(tf.float32, [None, num_input]) # Input images feedable
    #Y = tf.placeholder(tf.float32, [None, num_classes]) # Ground truth feedable
    Y = tf.placeholder(tf.float32, shape=(None,num_input*batch_size,1))
    logits = CNN(X) # Unscaled probabilities

    prediction = tf.nn.softmax(logits) # Class-wise probabilities

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()

    _batch_x = np.random.rand(batch_size,num_input)
    _batch_y = np.random.rand(1,num_input*batch_size,1)

    # Start training
from tensorflow.python import debug as tf_debug

with tf.Session(config=config) as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # Run the initializer
    sess.run(init)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)


    for step in range(100):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = _batch_x
        batch_y = _batch_y

            # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y}, options=run_options)

    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y : batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

    # Get test set accuracy
    #print("Testing Accuracy:",sess.run(accuracy, feed_dict={X: mnist.test.images[:256],Y: mnist.test.labels[:256]}))

