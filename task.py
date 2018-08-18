import tensorflow as tf
import argparse
FLAGS = None

parser = argparse.ArgumentParser()
# Flags for defining the tf.train.Server
parser.add_argument(
    "--task_index",
    type=int,
    default=1,
    help="Index of task within the job"
)

FLAGS, unparsed = parser.parse_known_args()

# Define your list of IP address / port number combos
IP_ADDRESS1='localhost'
PORT1='2222'
IP_ADDRESS2='localhost'
PORT2='2224'

# Define cluster
cluster_spec = tf.train.ClusterSpec({'worker' : [(IP_ADDRESS1 + ":" + PORT1), (IP_ADDRESS2 + ":" + PORT2)]})

# Define server for specific machine
task_index = FLAGS.task_index
server = tf.train.Server(cluster_spec, job_name='worker', task_index=task_index)

# Server will run as long as the notebook is running
server.join()

