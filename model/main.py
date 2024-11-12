# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:16:30 2020

@author: LSH
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 비활성화
import tensorflow as tf
import time
import datetime
import pytz
import platform
import psutil
import ray

from utils import DKT
from load_data import DKTData
from model import LayerNormBasicLSTMCell  # 새로 추가한 클래스 임포트

ray.shutdown()
ray.init(ignore_reinit_error=True, local_mode=True)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

rnn_cells = {
    "LSTM": tf.compat.v1.nn.rnn_cell.LSTMCell,
    "GRU": tf.compat.v1.nn.rnn_cell.GRUCell,
    "BasicRNN": tf.compat.v1.nn.rnn_cell.BasicRNNCell,
    "LayerNormBasicLSTM": LayerNormBasicLSTMCell,  # 수정된 부분
}

num_runs = 1
num_epochs = 25
batch_size = 4
keep_prob = 0.8656542586183774

network_config = {}
network_config['batch_size'] = batch_size
network_config['hidden_layer_structure'] = [24, ]
network_config['learning_rate'] = 0.004155923499457689
network_config['keep_prob'] = keep_prob
network_config['rnn_cell'] = rnn_cells['LayerNormBasicLSTM']  # 수정된 부분
network_config['max_grad_norm'] = 5.0
network_config['lambda_w1'] = 0.03
network_config['lambda_w2'] = 3.0
network_config['lambda_o'] = 0.1

train_path = './data/train_split_with_rows.csv'
valid_path = './data/valid_split_with_rows.csv'
test_path = './data/test_split_with_rows.csv'
save_dir_prefix = './results2/'
best_model_save_path = "./results2/model/LSTM-102"

def main():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    # 시스템 정보 출력
    local_tz = pytz.timezone('Asia/Seoul')
    date = datetime.datetime.now(datetime.timezone.utc)
    current_time = date.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
    print('Current time: {}, TensorFlow version: {}'.format(current_time, tf.__version__))
    print('OS information: {}'.format(platform.system()))
    print('OS version: {}'.format(platform.version()))
    print('Processor information: {}'.format(platform.processor()))
    print('CPU count: {}'.format(os.cpu_count()))
    print('RAM size: {}(GB)'.format(str(round(psutil.virtual_memory().total / (1024.0 ** 3)))))
    os.system('nvidia-smi')
    
    data = DKTData(train_path, valid_path, test_path, batch_size=batch_size)
    data_train = data.train
    data_valid = data.valid
    data_test = data.test
    num_problems = data.num_problems
    
    dkt = DKT(sess, data_train, data_valid, data_test, num_problems, network_config,
              save_dir_prefix=save_dir_prefix,
              num_runs=num_runs, num_epochs=num_epochs,
              keep_prob=keep_prob, logging=True, save=True, threshold=0.7)
    
    # 모델 최적화 실행
    dkt.model.build_graph()
    dkt.run_optimization()
    
    # 세션 종료
    sess.close()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Program run for: {:.2f}s".format(end_time - start_time))
