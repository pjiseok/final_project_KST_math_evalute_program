import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    seq_length = tf.reduce_sum(used, 1)
    seq_length = tf.cast(seq_length, tf.int32)
    return seq_length

class LayerNormBasicLSTMCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """Layer Normalized Basic LSTM Cell compatible with tf.compat.v1.nn.dynamic_rnn."""

    def __init__(self, num_units, forget_bias=1.0, activation=None, reuse=None, name=None):
        super(LayerNormBasicLSTMCell, self).__init__(_reuse=reuse, name=name)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or tf.math.tanh

        # LayerNormalization 인스턴스 생성
        self.layer_norm_i = tf.keras.layers.LayerNormalization()
        self.layer_norm_j = tf.keras.layers.LayerNormalization()
        self.layer_norm_f = tf.keras.layers.LayerNormalization()
        self.layer_norm_o = tf.keras.layers.LayerNormalization()
        self.layer_norm_c = tf.keras.layers.LayerNormalization()

        # Dense 레이어 생성
        self.dense = tf.keras.layers.Dense(4 * self._num_units, use_bias=True)

    @property
    def state_size(self):
        return tf.compat.v1.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        return tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
            tf.zeros([batch_size, self._num_units], dtype=dtype),
            tf.zeros([batch_size, self._num_units], dtype=dtype))

    def call(self, inputs, state):
        c_prev, h_prev = state
        inputs_and_h = tf.concat([inputs, h_prev], axis=1)

        # tf.layers.dense를 tf.keras.layers.Dense로 대체
        concat = self.dense(inputs_and_h)

        i, j, f, o = tf.split(concat, 4, axis=1)

        # Layer Normalization 적용
        i = self.layer_norm_i(i)
        j = self.layer_norm_j(j)
        f = self.layer_norm_f(f)
        o = self.layer_norm_o(o)

        g = self._activation(j)
        new_c = c_prev * tf.math.sigmoid(f + self._forget_bias) + tf.math.sigmoid(i) * g
        new_c = self.layer_norm_c(new_c)

        new_h = self._activation(new_c) * tf.math.sigmoid(o)

        new_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        return new_h, new_state

class Model(object):
    def __init__(self, num_problems,
                 hidden_layer_structure=(200,),
                 batch_size=32,
                 rnn_cell=LayerNormBasicLSTMCell,
                 learning_rate=0.01,
                 max_grad_norm=5.0,
                 lambda_w1=0.0,
                 lambda_w2=0.0,
                 lambda_o=0.0,
                 **kwargs):
        self.num_problems = num_problems
        self.hidden_layer_structure = hidden_layer_structure
        self.batch_size = batch_size
        self.rnn_cell = rnn_cell
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2
        self.lambda_o = lambda_o

    def _create_placeholder(self):
        print("Creating placeholder...")
        num_problems = self.num_problems
        self.X = tf.compat.v1.placeholder(tf.float32, [None, None, 2 * num_problems], name='X')
        self.y_seq = tf.compat.v1.placeholder(tf.float32, [None, None, num_problems], name='y_seq')
        self.y_corr = tf.compat.v1.placeholder(tf.float32, [None, None, num_problems], name='y_corr')
        self.keep_prob = tf.compat.v1.placeholder_with_default(1.0, shape=(), name='keep_prob')
        self.hidden_layer_input = self.X
        self.seq_length = length(self.X)

    def _influence(self):
        print("Creating RNN layers...")
        hidden_layer_structure = self.hidden_layer_structure
        self.hidden_layers_outputs = []
        self.hidden_layers_state = []
        hidden_layer_input = self.hidden_layer_input
        print("LSTM input shape: {0}".format(hidden_layer_input.get_shape()))
        for i, layer_state_size in enumerate(hidden_layer_structure):
            variable_scope_name = "hidden_layer_{}".format(i)
            with tf.compat.v1.variable_scope(variable_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
                cell = self.rnn_cell(num_units=layer_state_size)
                cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                outputs, state = tf.compat.v1.nn.dynamic_rnn(
                    cell,
                    hidden_layer_input,
                    dtype=tf.float32,
                    sequence_length=self.seq_length
                )
            self.hidden_layers_outputs.append(outputs)
            self.hidden_layers_state.append(state)
            hidden_layer_input = outputs

    def _create_loss(self):
        print("Creating Loss...")
        last_layer_size = self.hidden_layer_structure[-1]
        last_layer_outputs = self.hidden_layers_outputs[-1]
        with tf.compat.v1.variable_scope("output_layer", reuse=tf.compat.v1.AUTO_REUSE):
            W_yh = tf.compat.v1.get_variable("weights", shape=[last_layer_size, self.num_problems],
                                   initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.num_problems)))
            b_yh = tf.compat.v1.get_variable("biases", shape=[self.num_problems, ],
                                   initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.num_problems)))
            num_steps = tf.shape(last_layer_outputs)[1]
            self.outputs_flat = tf.reshape(last_layer_outputs, shape=[-1, last_layer_size])
            self.logits_flat = tf.matmul(self.outputs_flat, W_yh) + b_yh
            self.logits = tf.reshape(self.logits_flat, shape=[-1, num_steps, self.num_problems])
            self.preds = tf.sigmoid(self.logits, name="preds")
            target_indices = tf.where(tf.not_equal(self.y_seq, 0))
            self.target_logits = tf.gather_nd(self.logits, target_indices)
            self.target_preds = tf.gather_nd(self.preds, target_indices)
            self.target_labels = tf.gather_nd(self.y_corr, target_indices)
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.target_logits,
                                                                         labels=self.target_labels)
            self.loss = tf.reduce_mean(self.cross_entropy)
            current_seq = self.X[:, :, :self.num_problems]
            current_corr = self.X[:, :, self.num_problems:]
            self.target_indices_current = tf.where(tf.not_equal(current_seq, 0))
            self.target_logits_current = tf.gather_nd(self.logits, self.target_indices_current)
            self.target_preds_current = tf.gather_nd(self.preds, self.target_indices_current)
            self.target_labels_current = tf.gather_nd(current_corr, self.target_indices_current)
            self.cross_entropy_current = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.target_logits_current,
                                                                                 labels=self.target_labels_current)
            self.loss += self.lambda_o * tf.reduce_mean(self.cross_entropy_current)
            mask = length(self.y_seq)
            self.total_num_steps = tf.reduce_sum(tf.cast(mask, tf.float32))
            waviness_norm_l1 = tf.abs(self.preds[:, 1:, :] - self.preds[:, :-1, :])
            self.waviness_l1 = tf.reduce_sum(waviness_norm_l1) / self.total_num_steps / self.num_problems
            self.loss += self.lambda_w1 * self.waviness_l1
            waviness_norm_l2 = tf.square(self.preds[:, 1:, :] - self.preds[:, :-1, :])
            self.waviness_l2 = tf.reduce_sum(waviness_norm_l2) / self.total_num_steps / self.num_problems
            self.loss += self.lambda_w2 * self.waviness_l2

    def _create_optimizer(self):
        print('Creating optimizer...')
        with tf.compat.v1.variable_scope('Optimizer'):
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            gvs = self.optimizer.compute_gradients(self.loss)
            clipped_gvs = [(tf.clip_by_norm(grad, self.max_grad_norm), var) for grad, var in gvs]
            self.train_op = self.optimizer.apply_gradients(clipped_gvs)

    def _add_summary(self):
        pass

    def build_graph(self):
        self._create_placeholder()
        self._influence()
        self._create_loss()
        self._create_optimizer()
        self._add_summary()
