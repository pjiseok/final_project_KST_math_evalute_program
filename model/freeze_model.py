# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:50:02 2021

@author: LSH
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def generate_freezed_graph():
    output_node_names = 'output_layer/preds'
    output_graph = './results/model/frozen_model.pb'
    checkpoint_path = './results/model/LSTM-102'
    graph_path = './results/model/model.pb'

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        # 그래프 로드
        with tf.io.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        # 체크포인트 로드
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, checkpoint_path)

        # 변수들을 상수로 변환하여 그래프 동결
        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names.split(','))

        # 동결된 그래프 저장
        with tf.io.gfile.GFile(output_graph, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

if __name__ == "__main__":
    generate_freezed_graph()
