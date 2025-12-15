# cnn.py
import numpy as np

# ---------------------------------------------------------------------
# Activation functions and helper
# ---------------------------------------------------------------------

def relu(x):
    return np.maximum(0.0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def linear(x):
    return x

def get_activation(name):
    if name is None:
        return None
    name = name.lower()
    if name == "relu":
        return relu
    if name == "sigmoid":
        return sigmoid
    if name == "tanh":
        return tanh
    if name == "linear":
        return linear
    raise ValueError(f"Unknown activation: {name}")


# ---------------------------------------------------------------------
# Layers (all NumPy, NHWC format: (N, H, W, C))
# ---------------------------------------------------------------------

def conv2d_layer(x, W, b, stride=1, padding="same", activation=None):
    """
    x: (N, H, W, C_in)
    W: (kH, kW, C_in, C_out)
    b: (C_out,)
    Returns: (N, H_out, W_out, C_out)
    """
    N, H, W_in, C_in = x.shape
    kH, kW, C_in_W, C_out = W.shape
    assert C_in == C_in_W, "Input channels mismatch"

    if padding == "same":
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
    elif padding == "valid":
        pad_h = 0
        pad_w = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    x_padded = np.pad(
        x,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="constant"
    )

    H_out = (H + 2 * pad_h - kH) // stride + 1
    W_out = (W_in + 2 * pad_w - kW) // stride + 1

    out = np.zeros((N, H_out, W_out, C_out), dtype=np.float32)

    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                patch = x_padded[n,
                                 h_start:h_start + kH,
                                 w_start:w_start + kW,
                                 :]  # (kH, kW, C_in)
                # tensordot over (kH, kW, C_in) -> (C_out,)
                out[n, i, j, :] = np.tensordot(
                    patch, W, axes=([0, 1, 2], [0, 1, 2])
                ) + b

    if activation is not None:
        out = activation(out)

    return out


def pool2d_layer(x, pool_size=2, stride=2, mode="max"):
    """
    Simple 2D pooling layer (max or average), NHWC.
    x: (N, H, W, C)
    """
    N, H, W, C = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = np.zeros((N, H_out, W_out, C), dtype=x.dtype)

    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                patch = x[n,
                          h_start:h_start + pool_size,
                          w_start:w_start + pool_size,
                          :]  # (ps, ps, C)
                if mode == "max":
                    out[n, i, j, :] = np.max(patch, axis=(0, 1))
                elif mode == "avg":
                    out[n, i, j, :] = np.mean(patch, axis=(0, 1))
                else:
                    raise ValueError("mode must be 'max' or 'avg'")
    return out


def upsample2d_layer(x, scale=2):
    """
    Nearest-neighbor upsampling, NHWC.
    x: (N, H, W, C) -> (N, H*scale, W*scale, C)
    """
    N, H, W, C = x.shape
    H_out = H * scale
    W_out = W * scale

    out = np.zeros((N, H_out, W_out, C), dtype=x.dtype)

    for i in range(H_out):
        for j in range(W_out):
            out[:, i, j, :] = x[:, i // scale, j // scale, :]

    return out


def flatten_layer(x):
    """
    Flattens everything except the batch dimension.
    x: (N, ...) -> (N, D)
    """
    N = x.shape[0]
    return x.reshape(N, -1)


def dense_layer(x, W, b, activation=None):
    """
    Fully-connected layer.
    x: (N, D_in)
    W: (D_in, D_out)
    b: (D_out,)
    """
    out = x @ W + b
    if activation is not None:
        out = activation(out)
    return out


# ---------------------------------------------------------------------
# CNN class
# ---------------------------------------------------------------------

class CNN:
    """
    Generic CNN for image-to-image tasks.

    The model is defined by:
        - W_list: list of weight tensors (conv or dense), or None for layers
                  that have no trainable params (pool, upsample, flatten)
        - b_list: list of bias vectors (same indexing as W_list)
        - lname:  list of layer type strings, one of:
                  'conv', 'conv_out', 'pool', 'upsample', 'flatten', 'dense'

    Training (backprop and parameter updates) is handled in the notebook.
    """

    def __init__(self, dataset=None, verbose=False):
        self.dataset = dataset
        self.verbose = verbose
        self.W = None
        self.b = None
        self.lname = None
        self.activation_name = "relu"
        self.activation = relu

    def setup_model(self, W_list, b_list, lname, activation="relu"):
        """
        Store model parameters and structure.

        W_list, b_list, lname are parallel lists of the same length.
        """
        assert len(W_list) == len(b_list) == len(lname), \
            "W_list, b_list, and lname must have the same length"

        self.W = W_list
        self.b = b_list
        self.lname = lname
        self.activation_name = activation
        self.activation = get_activation(activation)

        if self.verbose:
            print("CNN model set up with layers:")
            for i, name in enumerate(self.lname):
                print(f"  Layer {i}: {name}, W shape: "
                      f"{None if self.W[i] is None else self.W[i].shape}")

    # -------------------------- forward pass -------------------------- #

    def feedforward_sample(self, x):
        """
        Forward pass for a single sample or a mini-batch.
        x: (N, H, W, C) or (N, D) depending on first layer.
        """
        a = x

        for i, name in enumerate(self.lname):
            if name == "conv":
                a = conv2d_layer(
                    a,
                    self.W[i],
                    self.b[i],
                    stride=1,
                    padding="same",
                    activation=self.activation
                )
            elif name == "conv_out":
                # final conv layer with linear activation (no non-linearity)
                a = conv2d_layer(
                    a,
                    self.W[i],
                    self.b[i],
                    stride=1,
                    padding="same",
                    activation=None
                )
            elif name == "pool":
                a = pool2d_layer(a, pool_size=2, stride=2, mode="max")
            elif name == "upsample":
                a = upsample2d_layer(a, scale=2)
            elif name == "flatten":
                a = flatten_layer(a)
            elif name == "dense":
                a = dense_layer(a, self.W[i], self.b[i],
                                activation=self.activation)
            else:
                raise ValueError(f"Unknown layer type in lname: {name}")

        return a

    def feedforward(self, x, batch_size=None):
        """
        Forward pass over a full dataset (N samples).

        If batch_size is None, processes all samples at once.
        """
        x = np.asarray(x)
        N = x.shape[0]

        if batch_size is None:
            return self.feedforward_sample(x)

        outputs = []
        for i in range(0, N, batch_size):
            xb = x[i:i + batch_size]
            yb = self.feedforward_sample(xb)
            outputs.append(yb)
        return np.concatenate(outputs, axis=0)

    # -------------------------- evaluation --------------------------- #

    def evaluate(self, x, y_true, metric="mse", batch_size=None, max_val=1.0):
        """
        Evaluate image-to-image performance.

        x:       inputs (N, H, W, C)
        y_true:  target images (N, H, W, C)
        metric:  'mse', 'mae', or 'psnr'
        max_val: value used to compute PSNR (e.g. 1.0 if inputs in [0, 1])

        Returns a dict with at least {'loss': <float>} and optionally 'psnr'.
        """
        y_pred = self.feedforward(x, batch_size=batch_size)
        y_true = np.asarray(y_true)

        if metric == "mse":
            loss = np.mean((y_pred - y_true) ** 2)
        elif metric == "mae":
            loss = np.mean(np.abs(y_pred - y_true))
        elif metric == "psnr":
            mse = np.mean((y_pred - y_true) ** 2)
            eps = 1e-8
            loss = mse  # we still report MSE as "loss"
            psnr = 10.0 * np.log10((max_val ** 2) / (mse + eps))
            return {"loss": float(loss), "psnr": float(psnr)}
        else:
            raise ValueError("metric must be 'mse', 'mae', or 'psnr'")

        return {"loss": float(loss)}


# ---------------------------------------------------------------------
# Helper: initialize an encoder-decoder image-to-image CNN
# ---------------------------------------------------------------------

def init_image_to_image_cnn(input_shape,
                            num_filters=(32, 64, 64),
                            num_output_channels=None,
                            rng=None):
    """
    Convenience function to build W_list, b_list, lname for an encoder-decoder.

    input_shape: (H, W, C_in)
    num_filters: sequence of conv filters for encoder (and decoder)
    num_output_channels: defaults to C_in if None
    rng: np.random.RandomState or None

    Returns: W_list, b_list, lname
    """
    if rng is None:
        rng = np.random.RandomState(123)

    H, W, C_in = input_shape
    if num_output_channels is None:
        C_out = C_in
    else:
        C_out = num_output_channels

    W_list = []
    b_list = []
    lname = []

    # ----- Encoder: [conv + pool] * len(num_filters) -----
    C_prev = C_in
    for nf in num_filters:
        # conv
        Wc = rng.randn(3, 3, C_prev, nf).astype(np.float32) * \
            np.sqrt(2.0 / (3 * 3 * C_prev))
        bc = np.zeros((nf,), dtype=np.float32)
        W_list.append(Wc)
        b_list.append(bc)
        lname.append("conv")

        # pool
        W_list.append(None)
        b_list.append(None)
        lname.append("pool")

        C_prev = nf

    # ----- Decoder: mirror except final layer -----
    # we won't upsample after the last encoder filter (bottleneck)
    decoder_filters = list(num_filters[:-1])[::-1]

    for nf in decoder_filters:
        # upsample
        W_list.append(None)
        b_list.append(None)
        lname.append("upsample")

        # conv
        Wc = rng.randn(3, 3, C_prev, nf).astype(np.float32) * \
            np.sqrt(2.0 / (3 * 3 * C_prev))
        bc = np.zeros((nf,), dtype=np.float32)
        W_list.append(Wc)
        b_list.append(bc)
        lname.append("conv")

        C_prev = nf

    # ----- Final conv to get desired output channels (no activation) -----
    Wc = rng.randn(3, 3, C_prev, C_out).astype(np.float32) * \
        np.sqrt(2.0 / (3 * 3 * C_prev))
    bc = np.zeros((C_out,), dtype=np.float32)
    W_list.append(Wc)
    b_list.append(bc)
    lname.append("conv_out")

    return W_list, b_list, lname
