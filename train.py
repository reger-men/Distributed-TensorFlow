import tensorflow as tf
import argparse
FLAGS = None

# Get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Some numbers
batch_size = 50
display_step = 1
num_input = 784
num_classes = 10


# Flags for defining the tf.train.Server
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_index",
    type=int,
    default=0,
    help="Index of task within the job"
)

FLAGS, unparsed = parser.parse_known_args()

# Set up tf session and initialize variables.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
config.gpu_options.allocator_type = 'BFC'

# cluster specification
parameter_servers = ["localhost:2222"]
workers = ["localhost:2223", "localhost:2224"]


def conv_layer(inputs, channels_in, channels_out, strides=1):

        # Create variables
        w=tf.Variable(tf.random_normal([3, 3, channels_in, channels_out]))
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
def CNN(x, devices):

    with tf.device(devices[1]):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1=conv_layer(x, 1, 3200, strides=1)
        pool1=maxpool2d(conv1)

        # Convolution Layer
        conv2=conv_layer(pool1, 3200, 64, strides=1)
        pool2=maxpool2d(conv2)

    with tf.device(devices[2]):
        # Fully connected layer
        fc1 = tf.reshape(pool2, [-1, 7*7*64])
        w1=tf.Variable(tf.random_normal([7*7*64, 1024]))
        b1=tf.Variable(tf.random_normal([1024]))
        fc1 = tf.add(tf.matmul(fc1,w1),b1)
        fc1=tf.nn.relu(fc1)

        # Output layer
        w2=tf.Variable(tf.random_normal([1024, num_classes]))
        b2=tf.Variable(tf.random_normal([num_classes]))
        out = tf.add(tf.matmul(fc1,w2),b2)

        # Check devices for good measure
        print(w1.device)
        print(b1.device)
        print(w2.device)
        print(b2.device)

    return out



# Define devices that we wish to split our graph over
device0='/job:ps/task:0'
device1='/job:worker/task:0'
device2='/job:worker/task:1'
devices=(device0, device1, device2)

tf.reset_default_graph() # Reset graph

# Construct model
with tf.device(devices[0]):
    X = tf.placeholder(tf.float32, [None, num_input]) # Input images feedable
    Y = tf.placeholder(tf.float32, [None, num_classes]) # Ground truth feedable

logits = CNN(X, devices) # Unscaled probabilities

with tf.device(devices[0]):

    prediction = tf.nn.softmax(logits) # Class-wise probabilities

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

# Define cluster
#cluster_spec = tf.train.ClusterSpec({'worker' : [(IP_ADDRESS1 + ":" + PORT1), (IP_ADDRESS2 + ":" + PORT2)]})
cluster_spec = tf.train.ClusterSpec({'ps' : parameter_servers, 'worker' : workers})

# Define server for specific machine
server = tf.train.Server(cluster_spec, job_name='worker', task_index=FLAGS.task_index, config=config)

# Check the server definition
server.server_def

# Start training
with tf.Session(server.target) as sess:
    # Run the initializer
    sess.run(init)

    for step in range(10000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y : batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

    # Get test set accuracy
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={X: mnist.test.images[:256],Y: mnist.test.labels[:256]}))
