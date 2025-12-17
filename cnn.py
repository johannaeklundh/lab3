# cnn.py
# Pure NumPy CNN utilities (NHWC) with forward + backward + simple SGD training.
# Keeps your list-of-layers structure and adds backprop + fit() so you can train >= 5 epochs.

import numpy as np
import csv
import os
import time

# ---------------------------------------------------------------------
# Activation functions and helpers
# ---------------------------------------------------------------------

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(z):
    # derivative w.r.t. pre-activation z
    return (z > 0).astype(z.dtype)

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

def get_activation_grad(name):
    """
    Derivative w.r.t. pre-activation (z), used in backprop.

    Implemented: 'relu', 'linear'.
    (You can add sigmoid/tanh if you need them, but ReLU + linear output is the standard for Proposal 2.)
    """
    if name is None:
        return None
    name = name.lower()
    if name == "relu":
        return relu_grad
    if name == "linear":
        return (lambda z: np.ones_like(z, dtype=z.dtype))
    raise ValueError(
        f"Activation grad for '{name}' is not implemented. "
        "Use activation='relu' and a linear output, or implement additional grads."
    )

# ---------------------------------------------------------------------
# Layers (all NumPy, NHWC format: (N, H, W, C))
# ---------------------------------------------------------------------

def _compute_same_padding(kH, kW):
    # Same padding convention consistent with your existing forward code
    return (kH - 1) // 2, (kW - 1) // 2

def conv2d_forward(x, W, b, stride=1, padding="same", activation=None):
    """
    x: (N, H, W, C_in)
    W: (kH, kW, C_in, C_out)
    b: (C_out,)
    Returns: out, cache
    """
    x = x.astype(np.float32, copy=False)
    W = W.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    N, H, W_in, C_in = x.shape
    kH, kW, C_in_W, C_out = W.shape
    if C_in != C_in_W:
        raise ValueError("Input channels mismatch")

    if padding == "same":
        pad_h, pad_w = _compute_same_padding(kH, kW)
    elif padding == "valid":
        pad_h = pad_w = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    x_padded = np.pad(
        x,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="constant"
    )

    H_out = (H + 2 * pad_h - kH) // stride + 1
    W_out = (W_in + 2 * pad_w - kW) // stride + 1

    z = np.zeros((N, H_out, W_out, C_out), dtype=np.float32)

    for n in range(N):
        for i in range(H_out):
            hs = i * stride
            for j in range(W_out):
                ws = j * stride
                patch = x_padded[n, hs:hs + kH, ws:ws + kW, :]  # (kH,kW,C_in)
                z[n, i, j, :] = np.tensordot(patch, W, axes=([0, 1, 2], [0, 1, 2])) + b

    out = activation(z) if activation is not None else z
    cache = (x, x_padded, W, b, stride, padding, pad_h, pad_w, z)
    return out, cache

def conv2d_backward(dout, cache, activation_name=None):
    """
    Backward pass for conv2d. Returns dx, dW, db.

    dout: gradient w.r.t. layer output (after activation if activation was used)
    activation_name: 'relu' for hidden conv layers, None/'linear' for output conv.
    """
    x, x_padded, W, b, stride, padding, pad_h, pad_w, z = cache
    N, H, W_in, C_in = x.shape
    kH, kW, _, C_out = W.shape
    _, H_out, W_out, _ = dout.shape

    # Convert dout to dz if activation applied
    if activation_name is None:
        dz = dout
    else:
        grad_fn = get_activation_grad(activation_name)
        dz = dout * grad_fn(z)

    dz = dz.astype(np.float32, copy=False)

    dW = np.zeros_like(W, dtype=np.float32)
    db = np.zeros_like(b, dtype=np.float32)
    dx_padded = np.zeros_like(x_padded, dtype=np.float32)

    # db: sum over N,H_out,W_out
    db[:] = np.sum(dz, axis=(0, 1, 2), dtype=np.float32)

    # dW and dx_padded
    for n in range(N):
        for i in range(H_out):
            hs = i * stride
            for j in range(W_out):
                ws = j * stride
                patch = x_padded[n, hs:hs + kH, ws:ws + kW, :]  # (kH,kW,C_in)
                # dW
                for co in range(C_out):
                    dW[:, :, :, co] += patch * dz[n, i, j, co]
                # dx_padded
                for co in range(C_out):
                    dx_padded[n, hs:hs + kH, ws:ws + kW, :] += W[:, :, :, co] * dz[n, i, j, co]

    # Unpad
    if pad_h == 0 and pad_w == 0:
        dx = dx_padded
    else:
        dx = dx_padded[:, pad_h:pad_h + H, pad_w:pad_w + W_in, :]
    return dx, dW, db

def pool2d_forward(x, pool_size=2, stride=2, mode="max"):
    """Simple pooling, NHWC. Returns out, cache."""
    N, H, W, C = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out = np.zeros((N, H_out, W_out, C), dtype=x.dtype)

    argmax = None
    if mode == "max":
        argmax = np.zeros((N, H_out, W_out, C, 2), dtype=np.int32)

    for n in range(N):
        for i in range(H_out):
            hs = i * stride
            for j in range(W_out):
                ws = j * stride
                patch = x[n, hs:hs + pool_size, ws:ws + pool_size, :]  # (ps,ps,C)
                if mode == "max":
                    flat = patch.reshape(pool_size * pool_size, C)
                    idx = np.argmax(flat, axis=0)
                    out[n, i, j, :] = flat[idx, np.arange(C)]
                    argmax[n, i, j, :, 0] = idx // pool_size
                    argmax[n, i, j, :, 1] = idx % pool_size
                elif mode == "avg":
                    out[n, i, j, :] = np.mean(patch, axis=(0, 1))
                else:
                    raise ValueError("mode must be 'max' or 'avg'")

    cache = (x, pool_size, stride, mode, argmax)
    return out, cache

def pool2d_backward(dout, cache):
    x, pool_size, stride, mode, argmax = cache
    N, H, W, C = x.shape
    _, H_out, W_out, _ = dout.shape
    dx = np.zeros_like(x, dtype=x.dtype)

    for n in range(N):
        for i in range(H_out):
            hs = i * stride
            for j in range(W_out):
                ws = j * stride
                if mode == "avg":
                    dx[n, hs:hs + pool_size, ws:ws + pool_size, :] += dout[n, i, j, :] / (pool_size * pool_size)
                else:
                    di = argmax[n, i, j, :, 0]
                    dj = argmax[n, i, j, :, 1]
                    for c in range(C):
                        dx[n, hs + di[c], ws + dj[c], c] += dout[n, i, j, c]
    return dx

def upsample2d_forward(x, scale=2):
    out = np.repeat(np.repeat(x, scale, axis=1), scale, axis=2)
    cache = (x.shape, scale)
    return out, cache

def upsample2d_backward(dout, cache):
    (N, H, W, C), scale = cache
    dx = np.zeros((N, H, W, C), dtype=dout.dtype)
    for i in range(H):
        for j in range(W):
            dx[:, i, j, :] = np.sum(
                dout[:, i*scale:(i+1)*scale, j*scale:(j+1)*scale, :],
                axis=(1, 2)
            )
    return dx

def flatten_forward(x):
    N = x.shape[0]
    out = x.reshape(N, -1)
    cache = x.shape
    return out, cache

def flatten_backward(dout, cache):
    return dout.reshape(cache)

def dense_forward(x, W, b, activation=None):
    z = x @ W + b
    out = activation(z) if activation is not None else z
    cache = (x, W, b, z)
    return out, cache

def dense_backward(dout, cache, activation_name=None):
    x, W, b, z = cache
    if activation_name is None:
        dz = dout
    else:
        grad_fn = get_activation_grad(activation_name)
        dz = dout * grad_fn(z)
    dW = x.T @ dz
    db = np.sum(dz, axis=0)
    dx = dz @ W.T
    return dx, dW, db

# ---------------------------------------------------------------------
# Loss (image-to-image)
# ---------------------------------------------------------------------

def mse_loss(y_pred, y_true):
    """Returns (loss, dL/dy_pred)."""
    y_pred = y_pred.astype(np.float32, copy=False)
    y_true = y_true.astype(np.float32, copy=False)
    diff = y_pred - y_true
    loss = np.mean(diff ** 2, dtype=np.float32)
    dyp = (2.0 / diff.size) * diff
    return float(loss), dyp

# ---------------------------------------------------------------------
# CNN class
# ---------------------------------------------------------------------

class CNN:
    """CNN for image-to-image tasks. Adds training support while keeping your layer-list structure."""

    def __init__(self, dataset=None, verbose=False):
        self.dataset = dataset
        self.verbose = verbose
        self.W = None
        self.b = None
        self.lname = None
        self.activation_name = "relu"
        self.activation = relu
        self._caches = None

    def setup_model(self, W_list, b_list, lname, activation="relu"):
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

    def forward(self, x, training=False):
        a = np.asarray(x)
        caches = [] if training else None

        for i, name in enumerate(self.lname):
            if name == "conv":
                a, cache = conv2d_forward(a, self.W[i], self.b[i],
                                         stride=1, padding="same",
                                         activation=self.activation)
                if training:
                    caches.append(("conv", cache))
            elif name == "conv_out":
                a, cache = conv2d_forward(a, self.W[i], self.b[i],
                                         stride=1, padding="same",
                                         activation=None)
                if training:
                    caches.append(("conv_out", cache))
            elif name == "pool":
                a, cache = pool2d_forward(a, pool_size=2, stride=2, mode="max")
                if training:
                    caches.append(("pool", cache))
            elif name == "upsample":
                a, cache = upsample2d_forward(a, scale=2)
                if training:
                    caches.append(("upsample", cache))
            elif name == "flatten":
                a, cache = flatten_forward(a)
                if training:
                    caches.append(("flatten", cache))
            elif name == "dense":
                a, cache = dense_forward(a, self.W[i], self.b[i],
                                         activation=self.activation)
                if training:
                    caches.append(("dense", cache))
            else:
                raise ValueError(f"Unknown layer type in lname: {name}")

        if training:
            self._caches = caches
        return a

    # Backwards-compatible API
    def feedforward_sample(self, x):
        return self.forward(x, training=False)

    def feedforward(self, x, batch_size=None):
        x = np.asarray(x)
        if batch_size is None:
            return self.forward(x, training=False)
        N = x.shape[0]
        outs = []
        for i in range(0, N, batch_size):
            outs.append(self.forward(x[i:i+batch_size], training=False))
        return np.concatenate(outs, axis=0)

    # -------------------------- backward pass ------------------------- #

    def backward(self, dout):
        if self._caches is None:
            raise RuntimeError("No caches available. Run forward(..., training=True) first.")

        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        for layer_idx in reversed(range(len(self.lname))):
            layer_type = self.lname[layer_idx]
            cache_type, cache = self._caches[layer_idx]
            if cache_type != layer_type:
                raise RuntimeError("Cache/layer mismatch (internal error).")

            if layer_type == "conv":
                dout, dW, db = conv2d_backward(dout, cache, activation_name=self.activation_name)
                grads_W[layer_idx] = dW
                grads_b[layer_idx] = db
            elif layer_type == "conv_out":
                dout, dW, db = conv2d_backward(dout, cache, activation_name=None)
                grads_W[layer_idx] = dW
                grads_b[layer_idx] = db
            elif layer_type == "pool":
                dout = pool2d_backward(dout, cache)
            elif layer_type == "upsample":
                dout = upsample2d_backward(dout, cache)
            elif layer_type == "flatten":
                dout = flatten_backward(dout, cache)
            elif layer_type == "dense":
                dout, dW, db = dense_backward(dout, cache, activation_name=self.activation_name)
                grads_W[layer_idx] = dW
                grads_b[layer_idx] = db
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        return grads_W, grads_b

    # -------------------------- training helpers ---------------------- #

    @staticmethod
    def _iter_minibatches(x, y, batch_size, shuffle=True, rng=None):
        x = np.asarray(x)
        y = np.asarray(y)
        N = x.shape[0]
        idx = np.arange(N)
        if shuffle:
            if rng is None:
                rng = np.random.RandomState(123)
            rng.shuffle(idx)
        for start in range(0, N, batch_size):
            bidx = idx[start:start+batch_size]
            yield x[bidx], y[bidx]

    @staticmethod
    def _clip(arr, c):
        if arr is None:
            return None
        return np.clip(arr, -c, c)

    def train_step(self, x_batch, y_batch, lr=1e-3, clip_grad=None):
        y_pred = self.forward(x_batch, training=True)
        loss, dyp = mse_loss(y_pred, y_batch)
        grads_W, grads_b = self.backward(dyp)

        # SGD update
        for i in range(len(self.lname)):
            if self.W[i] is None:
                continue
            gW = grads_W[i]
            gb = grads_b[i]
            if clip_grad is not None:
                gW = self._clip(gW, clip_grad)
                gb = self._clip(gb, clip_grad)
            self.W[i] = self.W[i] - lr * gW
            self.b[i] = self.b[i] - lr * gb

        return loss

    def fit(self,
            x_train, y_train,
            epochs=5,
            batch_size=16,
            lr=1e-3,
            shuffle=True,
            clip_grad=None,
            x_val=None, y_val=None,
            verbose=True,
            bar_len=25,
            csv_path=None,
            model_name="cnn"):
        
        """
        Train the model and optionally log epoch results to a CSV file.
        """

        history = {"train_loss": []}
        use_val = (x_val is not None and y_val is not None)
        if use_val:
            history["val_loss"] = []

        rng = np.random.RandomState(123)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        n_train = x_train.shape[0]
        total_steps = int(np.ceil(n_train / batch_size))

        # ---- CSV setup ----
        if csv_path is not None:
            write_header = not os.path.exists(csv_path)
            csv_file = open(csv_path, "a", newline="")
            csv_writer = csv.writer(csv_file)
            if write_header:
                csv_writer.writerow([
                    "model",
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "psnr",
                    "epoch_time_sec",
                    "learning_rate",
                    "batch_size"
                ])

        for ep in range(epochs):
            start_time = time.time()
            running_sum = 0.0
            steps = 0

            for step, (xb, yb) in enumerate(
                self._iter_minibatches(x_train, y_train, batch_size, shuffle=shuffle, rng=rng),
                start=1
            ):
                batch_loss = self.train_step(xb, yb, lr=lr, clip_grad=clip_grad)

                running_sum += batch_loss
                steps = step
                avg_loss = running_sum / steps

                if verbose:
                    frac = step / total_steps
                    filled = int(bar_len * frac)
                    bar = "=" * filled + ">" + "." * (bar_len - filled - 1)
                    pct = int(frac * 100)
                    print(
                        f"\rEpoch {ep+1}/{epochs} "
                        f"[{bar}] {pct:3d}%  "
                        f"batch_loss={batch_loss:.4e}  "
                        f"avg_loss={avg_loss:.4e}",
                        end="",
                        flush=True
                    )

            train_loss = running_sum / steps
            history["train_loss"].append(float(train_loss))

            if verbose:
                print()

            # ---- Validation ----
            val_loss = None
            psnr = None
            if use_val:
                y_pred = self.feedforward(x_val, batch_size=batch_size)
                val_loss, _ = mse_loss(y_pred, y_val)
                history["val_loss"].append(float(val_loss))

                mse = val_loss
                psnr = 10.0 * np.log10((1.0 ** 2) / (mse + 1e-8))

                if verbose:
                    print(f"           val_loss={val_loss:.4e}  psnr={psnr:.2f} dB")
            else:
                if verbose:
                    print(f"           train_loss={train_loss:.4e}")

            epoch_time = time.time() - start_time

            # ---- CSV write ----
            if csv_path is not None:
                csv_writer.writerow([
                    model_name,
                    ep + 1,
                    train_loss,
                    val_loss,
                    psnr,
                    epoch_time,
                    lr,
                    batch_size
                ])
                csv_file.flush()

        if csv_path is not None:
            csv_file.close()

        return history

    # -------------------------- evaluation --------------------------- #

    def evaluate(self, x, y_true, metric="mse", batch_size=None, max_val=1.0):
        y_pred = self.feedforward(x, batch_size=batch_size)
        y_true = np.asarray(y_true)

        if metric == "mse":
            loss = np.mean((y_pred - y_true) ** 2)
            return {"loss": float(loss)}
        elif metric == "mae":
            loss = np.mean(np.abs(y_pred - y_true))
            return {"loss": float(loss)}
        elif metric == "psnr":
            mse = np.mean((y_pred - y_true) ** 2)
            eps = 1e-8
            psnr = 10.0 * np.log10((max_val ** 2) / (mse + eps))
            return {"loss": float(mse), "psnr": float(psnr)}
        else:
            raise ValueError("metric must be 'mse', 'mae', or 'psnr'")

# ---------------------------------------------------------------------
# Helper: initialize an image-to-image CNN (conv stack)
# ---------------------------------------------------------------------

def init_deep_image_to_image_cnn(input_shape,
                                 num_filters=(32, 64, 64, 32),
                                 num_output_channels=None,
                                 rng=None):
    """
    Deeper image-to-image CNN:
        [conv -> conv -> conv -> conv -> conv_out]
    All convs use 'same' padding and stride=1 => H,W preserved.
    """
    if rng is None:
        rng = np.random.RandomState(123)

    H, W, C_in = input_shape
    C_out = C_in if num_output_channels is None else int(num_output_channels)

    W_list, b_list, lname = [], [], []
    C_prev = C_in

    # He init for ReLU layers
    for nf in num_filters:
        Wc = rng.randn(3, 3, C_prev, nf).astype(np.float32) * np.sqrt(2.0 / (3 * 3 * C_prev))
        bc = np.zeros((nf,), dtype=np.float32)
        W_list.append(Wc); b_list.append(bc); lname.append("conv")
        C_prev = nf

    # Output conv (linear)
    Wc = rng.randn(3, 3, C_prev, C_out).astype(np.float32) * np.sqrt(2.0 / (3 * 3 * C_prev))
    bc = np.zeros((C_out,), dtype=np.float32)
    W_list.append(Wc); b_list.append(bc); lname.append("conv_out")

    return W_list, b_list, lname

def _print_progress(epoch, epochs, step, total_steps, batch_loss, avg_loss, bar_len=25):
    frac = step / total_steps
    filled = int(bar_len * frac)
    bar = "=" * filled + ">" + "." * (bar_len - filled - 1)
    pct = int(frac * 100)

    msg = (
        f"\rEpoch {epoch}/{epochs} "
        f"[{bar}] {pct:3d}%  "
        f"batch_loss={batch_loss:.4e}  "
        f"avg_loss={avg_loss:.4e}"
    )
    print(msg, end="", flush=True)

