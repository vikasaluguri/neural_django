"""
Neural Network from Scratch — Pure NumPy Implementation
No TensorFlow, PyTorch, or sklearn for the core logic.
"""

import numpy as np
import json


# ─────────────────────────────────────────────
#  Activation Functions
# ─────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def softmax(z):
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu':    (relu,    relu_derivative),
    'tanh':    (tanh,    tanh_derivative),
}


# ─────────────────────────────────────────────
#  Loss Functions
# ─────────────────────────────────────────────

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def binary_cross_entropy(y_pred, y_true):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

LOSSES = {
    'mse': mse_loss,
    'binary_crossentropy': binary_cross_entropy,
}


# ─────────────────────────────────────────────
#  Neural Network Class
# ─────────────────────────────────────────────

class NeuralNetwork:
    """
    Fully-connected feedforward neural network built from scratch.

    Parameters
    ----------
    layer_sizes   : list of ints, e.g. [2, 4, 4, 1]
    activations   : list of str, one per hidden+output layer
    loss          : 'mse' | 'binary_crossentropy'
    learning_rate : float
    """

    def __init__(self, layer_sizes, activations, loss='mse', learning_rate=0.01):
        self.layer_sizes   = layer_sizes
        self.activations   = activations
        self.loss_name     = loss
        self.lr            = learning_rate
        self.loss_fn       = LOSSES[loss]
        self._init_params()
        self.history = {'loss': [], 'accuracy': []}

    def _init_params(self):
        np.random.seed(42)
        self.W, self.b = [], []
        for i in range(len(self.layer_sizes) - 1):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # He init for relu, Xavier for others
            if self.activations[i] == 'relu':
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.W.append(np.random.randn(fan_out, fan_in) * scale)
            self.b.append(np.zeros((fan_out, 1)))

    # ── Forward Pass ──────────────────────────
    def forward(self, X):
        """X shape: (features, samples)"""
        self._cache = {'A': [X], 'Z': []}
        A = X
        for i, act_name in enumerate(self.activations):
            Z = self.W[i] @ A + self.b[i]
            act_fn, _ = ACTIVATIONS[act_name]
            A = act_fn(Z)
            self._cache['Z'].append(Z)
            self._cache['A'].append(A)
        return A

    # ── Backward Pass ─────────────────────────
    def backward(self, Y):
        """Y shape: (output_size, samples)"""
        m   = Y.shape[1]
        grads_W, grads_b = [], []
        A_last = self._cache['A'][-1]
        # Output layer delta
        dA = (A_last - Y) / m

        for i in reversed(range(len(self.W))):
            _, act_deriv = ACTIVATIONS[self.activations[i]]
            Z  = self._cache['Z'][i]
            dZ = dA * act_deriv(Z)
            A_prev = self._cache['A'][i]
            dW = dZ @ A_prev.T
            db = np.sum(dZ, axis=1, keepdims=True)
            dA = self.W[i].T @ dZ
            grads_W.insert(0, dW)
            grads_b.insert(0, db)

        # Update weights
        for i in range(len(self.W)):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    # ── Training ──────────────────────────────
    def train(self, X, Y, epochs=1000, verbose_every=100):
        """
        X: (features, samples)   Y: (outputs, samples)
        Returns history dict.
        """
        self.history = {'loss': [], 'accuracy': []}
        for epoch in range(1, epochs + 1):
            Y_pred = self.forward(X)
            loss   = self.loss_fn(Y_pred, Y)
            self.backward(Y)

            # Accuracy (for binary classification)
            preds    = (Y_pred >= 0.5).astype(float)
            accuracy = float(np.mean(preds == Y) * 100)

            self.history['loss'].append(round(float(loss), 6))
            self.history['accuracy'].append(round(accuracy, 2))

        return self.history

    # ── Predict ───────────────────────────────
    def predict(self, X):
        """X: (features, samples) → returns probabilities"""
        return self.forward(X)

    def predict_class(self, X):
        probs = self.predict(X)
        return (probs >= 0.5).astype(int)

    # ── Serialise ─────────────────────────────
    def to_dict(self):
        return {
            'layer_sizes':   self.layer_sizes,
            'activations':   self.activations,
            'loss':          self.loss_name,
            'learning_rate': self.lr,
            'weights':       [w.tolist() for w in self.W],
            'biases':        [b.tolist() for b in self.b],
        }

    @classmethod
    def from_dict(cls, d):
        nn = cls(d['layer_sizes'], d['activations'], d['loss'], d['learning_rate'])
        nn.W = [np.array(w) for w in d['weights']]
        nn.b = [np.array(b) for b in d['biases']]
        return nn


# ─────────────────────────────────────────────
#  Built-in Datasets
# ─────────────────────────────────────────────

def dataset_xor():
    X = np.array([[0,0],[0,1],[1,0],[1,1]]).T   # (2, 4)
    Y = np.array([[0,1,1,0]])                    # (1, 4)
    return X, Y, ['Input A', 'Input B'], ['XOR Output']

def dataset_and():
    X = np.array([[0,0],[0,1],[1,0],[1,1]]).T
    Y = np.array([[0,0,0,1]])
    return X, Y, ['Input A', 'Input B'], ['AND Output']

def dataset_or():
    X = np.array([[0,0],[0,1],[1,0],[1,1]]).T
    Y = np.array([[0,1,1,1]])
    return X, Y, ['Input A', 'Input B'], ['OR Output']

def dataset_circles(n=200):
    """Two concentric circles — binary classification."""
    np.random.seed(0)
    angles = np.random.uniform(0, 2 * np.pi, n)
    r_inner = np.random.normal(1.0, 0.15, n // 2)
    r_outer = np.random.normal(2.5, 0.15, n // 2)
    r = np.concatenate([r_inner, r_outer])
    X = np.vstack([r * np.cos(angles), r * np.sin(angles)])
    Y = np.array([[0] * (n // 2) + [1] * (n // 2)])
    # Shuffle
    idx = np.random.permutation(n)
    return X[:, idx], Y[:, idx], ['x', 'y'], ['Class']

def dataset_moons(n=200):
    """Two interleaved half-moons."""
    np.random.seed(1)
    half = n // 2
    t0 = np.linspace(0, np.pi, half)
    t1 = np.linspace(np.pi, 2 * np.pi, half)
    X0 = np.vstack([np.cos(t0), np.sin(t0)]) + np.random.randn(2, half) * 0.1
    X1 = np.vstack([1 - np.cos(t1), 0.5 - np.sin(t1)]) + np.random.randn(2, half) * 0.1
    X  = np.hstack([X0, X1])
    Y  = np.array([[0] * half + [1] * half])
    idx = np.random.permutation(n)
    return X[:, idx], Y[:, idx], ['x', 'y'], ['Class']

DATASETS = {
    'xor':     dataset_xor,
    'and':     dataset_and,
    'or':      dataset_or,
    'circles': dataset_circles,
    'moons':   dataset_moons,
}
