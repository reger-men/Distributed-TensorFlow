import tensorflow as tf
import argparse
FLAGS = None

parser = argparse.ArgumentParser()
# Flags for defining the tf.train.Server
parser.add_argument(
    "--task_index",
    type=int,
    default=0,
    help="Index of task within the job"
)
parser.add_argument(
      "--job_name",
      type=str,
      default="ps",
      help="One of 'ps', 'worker'"
  )


FLAGS, unparsed = parser.parse_known_args()

# Set up tf session and initialize variables.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
config.gpu_options.allocator_type = 'BFC'

# cluster specification
parameter_servers = ["localhost:2222"]
workers = ["localhost:2223", "localhost:2224", "localhost:2225"]

# Define cluster
#cluster_spec = tf.train.ClusterSpec({'worker' : [(IP_ADDRESS1 + ":" + PORT1), (IP_ADDRESS2 + ":" + PORT2)]})
cluster_spec = tf.train.ClusterSpec({'ps' : parameter_servers, 'worker' : workers})

# Define server for specific machine
server = tf.train.Server(cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index, config=config)

# Server will run as long as the notebook is running
server.join()

