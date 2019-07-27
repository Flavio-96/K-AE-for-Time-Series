import tensorflow as tf
from tensorflow.keras.layers import LSTMCell
import tensorflow_probability as tfp


class Model:
    def __init__(self, config):
        # Hyperparameters of the net
        num_layers = config['num_layers']
        hidden_size = config['hidden_size']
        max_grad_norm = config['max_grad_norm']
        batch_size = config['batch_size']
        seq_length = config['seq_length']
        crd = config['crd']
        num_l = config['num_l']
        learning_rate = config['learning_rate']
        self.seq_length = seq_length
        self.batch_size = batch_size

        # Nodes for the input variables
        self.x = tf.placeholder("float", shape=[batch_size, seq_length], name='Input_data')
        self.x_exp = tf.expand_dims(self.x, 1)
        self.keep_prob = 1 - dropout

        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            # The encoder cell, multi-layered with dropout
            # Number of LSTM = hidden layer size
            cell_enc = tf.keras.layers.StackedRNNCells([
                tf.keras.layers.LSTMCell(
                    hidden_size,
                    dropout=self.keep_prob
                ) for _ in range(num_layers)
            ])
            # Initial state, tuple for all lstms stacked
            # layer for mean of z
            W_mu = tf.get_variable('W_mu', [hidden_size, num_l])

            # Creates a recurrent neural network specified by RNNCell cell
            # outputs is a length T list of outputs (one for each input), or a nested tuple of such elements.
            # in our case one output for each time series in input
            stacked_layer = tf.keras.layers.RNN(cell_enc, unroll=True)

            cell_output = stacked_layer(self.x_exp)
            b_mu = tf.get_variable('b_mu', [num_l])

            # self.z_mu is the Tensor containing the hidden representations
            # It can be used to do visualization, clustering or subsequent classification
            # tf.nn.xw_plus_b computes matmul(x, weights) + biases.
            self.z_mu = tf.nn.xw_plus_b(cell_output, W_mu, b_mu, name='z_mu')  # mu, mean, of latent space

            # Calculate the mean and variance of the latent space
            # The mean and variance are calculated by aggregating the contents of z_mu across axes
            lat_mean, lat_var = tf.nn.moments(self.z_mu, axes=[1])

            # Train the point in latent space to have zero-mean and unit-variance on batch basis
            self.loss_lat_batch = tf.reduce_mean(tf.square(lat_mean) + lat_var - tf.log(lat_var) - 1)

        with tf.name_scope("Lat_2_dec"):
            # layer to generate initial state
            W_state = tf.get_variable('W_state', [num_l, hidden_size])
            b_state = tf.get_variable('b_state', [hidden_size])
            z_state = tf.nn.xw_plus_b(self.z_mu, W_state, b_state, name='z_state')  # mu, mean, of latent space

        # Similar steps as encoder
        with tf.variable_scope("Decoder"):
            # The decoder, also multi-layered
            cell_dec = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_size) for _ in range(num_layers)])

            # Initial state
            initial_state_dec = tuple([(z_state, z_state)] * num_layers)
            dec_inputs = [tf.zeros([batch_size, 1])] * seq_length

            outputs_dec, _ = tf.contrib.rnn.static_rnn(cell_dec,
                                                       inputs=dec_inputs,
                                                       initial_state=initial_state_dec)
        with tf.name_scope("Out_layer"):
            params_o = 2 * crd  # Number of coordinates + variances
            W_o = tf.get_variable('W_o', [hidden_size, params_o])
            b_o = tf.get_variable('b_o', [params_o])
            outputs = tf.concat(outputs_dec, axis=0)  # tensor in [seq_length*batch_size,hidden_size]
            h_out = tf.nn.xw_plus_b(outputs, W_o, b_o)
            h_mu, h_sigma_log = tf.unstack(tf.reshape(h_out, [seq_length, batch_size, params_o]), axis=2)
            h_sigma = tf.exp(h_sigma_log)
            dist = tfp.distributions.Normal(h_mu, h_sigma)
            px = dist.log_prob(tf.transpose(self.x))
            loss_seq = -px
            self.loss_seq = tf.reduce_mean(loss_seq)

        with tf.name_scope("train"):
            global_step = tf.Variable(0, trainable=False)
            # Use learning rate decay
            # Useful use a learning rate schedule to reduce learning rate as the training progresses.
            lr = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.1, staircase=False)

            self.loss = self.loss_seq + self.loss_lat_batch

            # Route the gradients
            tvars = tf.trainable_variables()
            # We clip the gradients to prevent explosion
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
            self.numel = tf.constant([[0]])

            # And apply the gradients
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = zip(grads, tvars)
            self.train_step = optimizer.apply_gradients(gradients, global_step=global_step)

            self.numel = tf.constant([[0]])

        tf.summary.tensor_summary('lat_state', self.z_mu)
        # Define one op to call all summaries
        self.merged = tf.summary.merge_all()
        # Returns an Op that initializes global variables.
        self.init_op = tf.global_variables_initializer()
