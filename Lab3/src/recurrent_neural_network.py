import numpy as np
import dataset as ds

class RecurrentNeuralNetwork:
    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.randn(hidden_size, vocab_size) * 1e-2
        self.W = np.random.randn(hidden_size, hidden_size) * 1e-2
        self.b = np.zeros((hidden_size, 1))
        self.V = np.random.randn(vocab_size, hidden_size) * 1e-2
        self.c = np.zeros((vocab_size, 1))

        self.memory_U = np.zeros_like(self.U)
        self.memory_W = np.zeros_like(self.W)
        self.memory_b = np.zeros_like(self.b)
        self.memory_V = np.zeros_like(self.V)
        self.memory_c = np.zeros_like(self.c)

    def rnn_step_forward(self, X, h_prev):
        '''
        A single time step forward of a recurrent neural network with a hyperbolic tangent nonlinearity.

            x      - input data            (minibatch size x input dimension)
            h_prev - previous hidden state (minibatch size x hidden size)
        '''

        h = np.tanh(np.dot(h_prev, self.W.transpose())
                    + np.dot(X, self.U.transpose())
                    + self.b.transpose())
        cache = { 'h': h, 'h_prev': h_prev, 'X': X }
        return h, cache

    def rnn_forward(self, X, h0):
        '''
        Full unroll forward of the recurrent neural network with a hyperbolic tangent nonlinearity.
        
            X  - input data for the whole time-series (minibatch size x sequence_length x input dimension)
            h0 - initial hidden state                 (minibatch size x hidden size)
        '''

        def run_forward(X, h0):
            h = h0
            for x in np.transpose(X, (1, 0, 2)):
                h, cache = self.rnn_step_forward(x, h)
                yield h, cache

        H, Cache = zip(*run_forward(X, h0))
        return np.array(H), Cache

    def rnn_step_backward(self, grad_next, cache):
        '''
        A single time step backward of a recurrent neural network with a hyperbolic tangent nonlinearity.

            grad_next - upstream gradient of the loss with respect to the next hidden state and current output
            cache     - cached information from the forward pass
        '''

        h = cache['h']
        h_prev = cache['h_prev']
        X = cache['X']
        dtanh = 1 - h ** 2
        da = grad_next * dtanh
        dh_prev = np.dot(da, self.W)  # W bez .transpose()
        dU = np.dot(da.transpose(), X)
        dW = np.dot(da.transpose(), h_prev)
        db = da.sum(axis=0, keepdims=True).transpose()

        return dh_prev, dU, dW, db

    def rnn_backward(self, dh, Cache):
        '''Full unroll forward of the recurrent neural network with a hyperbolic tangent nonlinearity.'''

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

        dU = np.clip(dU, -5, 5)
        dW = np.clip(dW, -5, 5)
        db = np.clip(db, -5, 5)

        return dU, dW, db

    def output(self, H):
        return softmax(np.dot(H, self.V.transpose())
                       + self.c.transpose())

    def outputs_loss_and_grads(self, H, y):
        y_hat = self.output(H)
        diff_yhat_y = y_hat - y

        loss = -np.sum(np.log(y_hat[y == 1]))
        dh = np.dot(diff_yhat_y, self.V)
        dV = np.tensordot(diff_yhat_y, H, [(0, 1), (0, 1)])
        dc = diff_yhat_y.sum(axis=(0, 1)).reshape(-1, 1)

        #dh = np.clip(dh, -5, 5)  # ?
        dV = np.clip(dV, -5, 5)
        dc = np.clip(dc, -5, 5)

        return loss, dh, dV, dc

    def update(self, dU, dW, db, dV, dc):

        self.memory_U += dU ** 2 
        self.memory_W += dW ** 2 
        self.memory_b += db ** 2 
        self.memory_V += dV ** 2 
        self.memory_c += dc ** 2 

        delta = 1e-7

        self.U -= self.learning_rate * dU / (delta + np.sqrt(self.memory_U))
        self.W -= self.learning_rate * dW / (delta + np.sqrt(self.memory_W))
        self.b -= self.learning_rate * db / (delta + np.sqrt(self.memory_b))
        self.V -= self.learning_rate * dV / (delta + np.sqrt(self.memory_V))
        self.c -= self.learning_rate * dc / (delta + np.sqrt(self.memory_c))

    def sample(self, seed, n_sample):
        h0 = np.zeros((1, self.hidden_size))
        H, _ = self.rnn_forward(seed, h0)
        h0 = H[-1]
        sample = []
        o = self.output(h0[np.newaxis, :, :])
        y = np.argmax(o)
        sample.append(y)
        for _ in range(n_sample):
            h0, _ = self.rnn_step_forward(o[0], h0)
            o = self.output(h0[np.newaxis, :, :])
            y = np.argmax(o)
            sample.append(y)
        return sample

    def step(self, h_prev, X, Y):
        H, Cache = self.rnn_forward(X, h_prev)
        y = Y.transpose((1, 0, 2))
        loss, dh, dV, dc = self.outputs_loss_and_grads(H, y)
        dU, dW, db = self.rnn_backward(dh, Cache)
        self.update(dU, dW, db, dV, dc)
        return loss, H[-1, :, :]


def softmax(x):
    '''Compute softmax values for each sets of scores in x.'''
    return (np.exp(x.transpose()) / np.sum(np.exp(x.transpose()), axis=0)).transpose()

def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=70, learning_rate=0.2, sample_every=20):

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
            current_epoch += 1
            print('epoch: #{}, average loss: {}'.format(current_epoch, total_loss / dataset.size))
            seed = "have"

            sample = RNN.sample(seed=dataset.to_one_hot(dataset.encode(seed).reshape(1, -1)), n_sample=100)
            print('sample:\n{}\n'.format(seed + ''.join(dataset.decode(sample))))

            sample = RNN.sample(seed=dataset.to_one_hot(dataset.encode(' ').reshape(1, -1)), n_sample=100)
            print('random sample:\n{}\n'.format(''.join(dataset.decode(sample))))

            h0 = np.zeros((batch_size, hidden_size))
            total_loss = 0

        X_oh = dataset.to_one_hot(X)
        Y_oh = dataset.to_one_hot(Y)

        loss, h0 = RNN.step(h0, X_oh, Y_oh)
        total_loss += loss

        batch += 1