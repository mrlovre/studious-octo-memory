import numpy as np
import dataset as ds

class RecurrentNeuralNetwork:
    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        # input projection
        self.U = np.random.randn(hidden_size, vocab_size) * 1e-2
        # hidden-to-hidden projection
        self.W = np.random.randn(hidden_size, hidden_size) * 1e-2
        # input bias
        self.b = np.zeros((hidden_size, 1))
        # output projection
        self.V = np.random.randn(vocab_size, hidden_size) * 1e-2
        # output bias
        self.c = np.zeros((vocab_size, 1))

        # memory of past gradients - rolling sum of squares (for Adagrad)
        self.memory_U = np.zeros_like(self.U)
        self.memory_W = np.zeros_like(self.W)
        self.memory_b = np.zeros_like(self.b)
        self.memory_V = np.zeros_like(self.V)
        self.memory_c = np.zeros_like(self.c)

    def rnn_step_forward(self, X, h_prev, U, W, b):
        '''
        A single time step forward of a recurrent neural network with a hyperbolic tangent nonlinearity.

            x - input data (minibatch size x input dimension)
            h_prev - previous hidden state (minibatch size x hidden size)
            U - input projection matrix (input dimension x hidden size)
            W - hidden to hidden projection matrix (hidden size x hidden size)
            b - bias of shape (hidden size x 1)
        '''

        H_current = np.tanh(np.dot(h_prev, W.transpose()) + np.dot(X, U.transpose()) + b.transpose())
        # return the new hidden state and a tuple of values needed for the backward step
        cache = { 'h': H_current, 'h_prev': h_prev, 'X': X }
        return H_current, cache

    def rnn_forward(self, X, h0, U, W, b):
        '''
        Full unroll forward of the recurrent neural network with a hyperbolic tangent nonlinearity

            X - input data for the whole time-series (minibatch size x sequence_length x input dimension)
            h0 - initial hidden state (minibatch size x hidden size)
            U - input projection matrix (input dimension x hidden size)
            W - hidden to hidden projection matrix (hidden size x hidden size)
            b - bias of shape (hidden size x 1)
        '''

        def run_forward(X, h0):
            h = h0
            for x in np.transpose(X, (1, 0, 2)):
                h, cache = self.rnn_step_forward(x, h, U, W, b)
                yield h, cache

        H, Cache = zip(*run_forward(X, h0))

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

        return H, Cache

    def rnn_step_backward(self, grad_next, cache):
        '''
        A single time step backward of a recurrent neural network with a hyperbolic tangent nonlinearity.

            grad_next - upstream gradient of the loss with respect to the next hidden state and current output
            cache - cached information from the forward pass
        '''

        h = cache['h']
        h_prev = cache['h_prev']
        X = cache['X']
        dtanh = 1 - h ** 2
        da = (grad_next * dtanh)
        dh_prev = np.dot(self.W, da.transpose()).transpose()
        dU = np.dot(da.transpose(), X)
        dW = np.dot(da.transpose(), h_prev)
        db = da.sum(axis=0, keepdims=True).transpose()

        return dh_prev, dU, dW, db


    def rnn_backward(self, dh, Cache):
        '''
         Full unroll forward of the recurrent neural network with a hyperbolic tangent nonlinearity
        '''

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        grad_next = np.zeros(dh.shape[1:])
        T = len(Cache)
        for t in reversed(range(T)):
            cache = Cache[t]
            grad_next += dh[t]
            grad_next, dU_i, dW_i, db_i = self.rnn_step_backward(grad_next, cache)
            dU += dU_i
            dW += dW_i
            db += db_i

#         dU *= 1e-5
#         dW *= 1e-5
#         db *= 1e-5
        dU = np.clip(dU, -5, 5)
        dW = np.clip(dW, -5, 5)
        db = np.clip(db, -5, 5)

        return dU, dW, db

    def output(self, h, V, c):
        return softmax(np.dot(V, h) + c)

    def outputs_loss_and_grads(self, H, V, c, y):
        outputs = (np.dot(V, H).transpose()[:, :, :, np.newaxis] + c).reshape(y.shape)
        y_hat = softmax(outputs.transpose()).transpose()
        diff_yhat_y = y_hat - y

        loss = -np.sum(np.log(y_hat[y == 1]))
        dh = np.dot(diff_yhat_y, V)
        dV = np.tensordot(H, diff_yhat_y, [(0, 2), (0, 1)]).transpose()
        dc = diff_yhat_y.sum(axis=(0, 1)).reshape(-1, 1)

#         dV *= 1e-5
#         dc *= 1e-5

        dV = np.clip(dV, -5, 5)
        dc = np.clip(dc, -5, 5)

        return loss, dh, dV, dc

    def update(self, dU, dW, db, dV, dc,
                     U, W, b, V, c,
                     memory_U, memory_W, memory_b, memory_V, memory_c):

        # update memory matrices
        # perform the Adagrad update of parameters
        learning_rate = self.learning_rate

        self.memory_U = memory_U + dU ** 2
        self.memory_W = memory_W + dW ** 2
        self.memory_b = memory_b + db ** 2
        self.memory_V = memory_V + dV ** 2
        self.memory_c = memory_c + dc ** 2

        delta = 1e-7

        self.U = U - learning_rate * dU / (delta + np.sqrt(self.memory_U)) * 1
        self.W = W - learning_rate * dW / (delta + np.sqrt(self.memory_W)) * 1
        self.b = b - learning_rate * db / (delta + np.sqrt(self.memory_b)) * 1
        self.V = V - learning_rate * dV / (delta + np.sqrt(self.memory_V)) * 0.1
        self.c = c - learning_rate * dc / (delta + np.sqrt(self.memory_c)) * 0.1

    def sample(self, seed, n_sample):
        # inicijalizirati h0 na vektor nula
        # seed string je pretvoren u one_hot
        h0 = np.zeros((1, self.hidden_size))
        for x in seed:
            H, _ = self.rnn_forward(x.reshape((1,) + x.shape), h0, self.U, self.W, self.b)
        h0 = H[-1]
        sample = []
        o = softmax(self.output(h0.transpose(), self.V, self.c))
        for _ in range(n_sample):
            h0, _ = self.rnn_step_forward(o.transpose(), h0, self.U, self.W, self.b)
            o = softmax(self.output(h0.transpose(), self.V, self.c))
            y = np.argmax(o)
            sample.append(y)
        return sample

    def step(self, h_prev, X, Y):
        H, Cache = self.rnn_forward(X, h_prev, self.U, self.W, self.b)
        h0 = np.array(H).transpose((0, 2, 1))
        y = Y.transpose((1, 0, 2))
        loss, dh, dV, dc = self.outputs_loss_and_grads(h0, self.V, self.c, y)
        dU, dW, db = self.rnn_backward(dh, Cache)
        self.update(dU, dW, db, dV, dc,
                    self.U, self.W, self.b, self.V, self.c,
                    self.memory_U, self.memory_W, self.memory_b, self.memory_V, self.memory_c)
        return loss, h0.transpose()[:, :, -1]

def run_language_model(dataset, max_epochs, hidden_size=200, sequence_length=10, learning_rate=1e-1, sample_every=20):
    vocab_size = len(dataset.sorted_chars)
    RNN = RecurrentNeuralNetwork(hidden_size, sequence_length, vocab_size, learning_rate)

    current_epoch = 0
    batch = 0
    total_loss = 0
    batch_size = 50

    h0 = np.zeros((batch_size, hidden_size))

    minibatches_iterator = ds.minibatches_iterator(dataset.create_minibatches(batch_size, sequence_length))
    while current_epoch < max_epochs:
        e, X, Y = next(minibatches_iterator)

        if e and batch != 0:
            print('epoch: #{}, average loss: {}'.format(current_epoch, total_loss / dataset.size))
            sample = RNN.sample(seed=dataset.to_one_hot(dataset.encode("BERNICE:\nThat's a").reshape(1, -1)), n_sample=20)
            print('sample: {}\n'.format(''.join(dataset.decode(sample))))
            current_epoch += 1
            h0 = np.zeros((batch_size, hidden_size))
                # why do we reset the hidden state here?
            total_loss = 0

        # One-hot transform the X and Y batches
        X_oh = dataset.to_one_hot(X)
        Y_oh = dataset.to_one_hot(Y)

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = RNN.step(h0, X_oh, Y_oh)
        total_loss += loss

        batch += 1

def softmax(x):
    '''
    Compute softmax values for each sets of scores in x.
    '''
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# from main import main
# main('../data/selected_conversations.txt')
