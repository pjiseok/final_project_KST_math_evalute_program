import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import DKT
from load_data import DKTData
from model import LayerNormBasicLSTMCell  # 추가된 클래스 임포트

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

# 하이퍼파라미터 설정
rnn_cells = {
    "LSTM": tf.compat.v1.nn.rnn_cell.LSTMCell,
    "GRU": tf.compat.v1.nn.rnn_cell.GRUCell,
    "BasicRNN": tf.compat.v1.nn.rnn_cell.BasicRNNCell,
    "LayerNormBasicLSTM": LayerNormBasicLSTMCell,
}
batch_size = 4
keep_prob = 0.8656542586183774

network_config = {
    'batch_size': batch_size,
    'hidden_layer_structure': [102],
    'learning_rate': 0.004155923499457689,
    'keep_prob': keep_prob,
    'rnn_cell': rnn_cells['LayerNormBasicLSTM'],
    'max_grad_norm': 5.0,
    'lambda_w1': 0.03,
    'lambda_w2': 3.0,
    'lambda_o': 0.1
}

# 경로 설정
train_path = './data/train_split_with_rows.csv'
valid_path = './data/valid_split_with_rows.csv'
test_path = './data/test_split_with_rows.csv'
best_model_save_path = "./save_model/LayerNormBasicLSTM-102"

# 혼동 행렬 시각화 함수
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # 데이터 로드
    data = DKTData(train_path, valid_path, test_path, batch_size=batch_size)
    data_train = data.train
    data_valid = data.valid
    data_test = data.test
    num_problems = data.num_problems

    # DKT 모델 설정 및 평가
    dkt = DKT(sess, data_train, data_valid, data_test, num_problems, network_config,
              save_dir_prefix="./results/",
              num_runs=1, num_epochs=1,
              keep_prob=keep_prob, logging=True, save=True, threshold=0.7)

    dkt.model.build_graph()

    # 첫 번째 배치의 입력 데이터 크기 확인
    X_batch, y_seq_batch, y_corr_batch = data_test.next_batch()
    print(f"First batch input data shape: {X_batch.shape}, Labels shape: {y_corr_batch.shape}")

    # 시퀀스 길이 설정 (명시적으로 지정)
    sequence_length = X_batch.shape[1]
    print(f"Sequence length: {sequence_length}")

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess=sess, save_path=best_model_save_path)

    # 테스트 데이터에 대한 평가 수행
    y_true, y_pred = dkt.evaluate('test', makefile=False)

    # 혼동 행렬 시각화
    plot_confusion_matrix(y_true, y_pred)
