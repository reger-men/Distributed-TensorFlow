import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import time

from model import PSPNet50
from tools import prepare_label
from image_reader import ImageReader

# Saving memory using gradient-checkpoint
#from gradient.memory_saving_gradients import gradients


IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

display_step = 1
BATCH_SIZE = 2
DATA_DIRECTORY = '/SSD_data/cityscapes_dataset/cityscape'
DATA_LIST_PATH = './list/cityscapes_train_list.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '713,713'
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 60001
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
RESTORE_FROM = './'
SNAPSHOT_DIR = './model/'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 50

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")
    return parser.parse_args()



# Get Args values
args = get_arguments()

# Set up tf session and initialize variables.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
config.gpu_options.allocator_type = 'BFC'

# cluster specification
parameter_servers = ["localhost:2222"]
workers = ["localhost:2223", "localhost:2224", "localhost:2225"]

# Define devices that we wish to split our graph over
device0='/job:ps/task:0'
device1='/job:worker/task:0'
device2='/job:worker/task:1'
device3='/job:worker/task:2'
devices=(device0, device1, device2, device3)

tf.reset_default_graph() # Reset graph

# Construct model
h, w = map(int, args.input_size.split(','))
input_size = (h, w)

tf.set_random_seed(args.random_seed)

coord = tf.train.Coordinator()

with tf.name_scope("create_inputs"):
    reader = ImageReader(
        args.data_dir,
        args.data_list,
        input_size,
        args.random_scale,
        args.random_mirror,
        args.ignore_label,
        IMG_MEAN,
        coord)
    image_batch, label_batch = reader.dequeue(args.batch_size)

net = PSPNet50({'data': image_batch}, is_training=True, num_classes=args.num_classes, devices=devices)
raw_output = net.layers['conv6']

with tf.device(devices[2]):
    # According from the prototxt in Caffe implement, learning rate must multiply by 10.0 in pyramid module
    fc_list = ['conv5_3_pool1_conv', 'conv5_3_pool2_conv', 'conv5_3_pool3_conv', 'conv5_3_pool6_conv', 'conv6', 'conv5_4']
    restore_var = [v for v in tf.global_variables()]
    all_trainable = [v for v in tf.trainable_variables() if ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]
    fc_trainable = [v for v in all_trainable if v.name.split('/')[0] in fc_list]
    conv_trainable = [v for v in all_trainable if v.name.split('/')[0] not in fc_list] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

    # Using Poly learning rate policy
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))


    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
        opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
        opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

        grads = gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
        #grads = gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable, checkpoints=[net.layers['conv2_3/relu'], net.layers['conv3_4/relu'], net.layers['conv4_2/relu'], net.layers['conv4_6/relu'], net.layers['conv5_3/relu']])
        grads_conv = grads[:len(conv_trainable)]
        grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
        grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

        train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
        train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
        train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

        train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)


    init = tf.global_variables_initializer()

# Define cluster
#cluster_spec = tf.train.ClusterSpec({'worker' : [(IP_ADDRESS1 + ":" + PORT1), (IP_ADDRESS2 + ":" + PORT2)]})
cluster_spec = tf.train.ClusterSpec({'ps' : parameter_servers, 'worker' : workers})

# Define server for specific machine
server = tf.train.Server(cluster_spec, job_name='worker', task_index=args.task_index, config=config)

# Check the server definition
server.server_def

# Start training
with tf.Session(server.target) as sess:
    # Run the initializer
    sess.run(init)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = {step_ph: step}

        # Run optimization op (backprop)
        loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)

        duration = time.time() - start_time

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            #loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y : batch_y})
            #print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

coord.request_stop()
coord.join(threads)
