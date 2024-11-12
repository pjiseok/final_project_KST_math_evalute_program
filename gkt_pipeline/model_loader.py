# model_loader.py
import tensorflow as tf

def load_checkpoint_model(checkpoint_dir):
    sess = tf.compat.v1.Session()
    checkpoint_path = checkpoint_dir
    saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + ".meta")
    saver.restore(sess, checkpoint_path)
    graph = tf.compat.v1.get_default_graph()
    return sess, graph

def get_model_tensors(graph):
    input_tensor = graph.get_tensor_by_name('X:0')
    output_tensor = graph.get_tensor_by_name('output_layer/preds:0')
    return input_tensor, output_tensor
