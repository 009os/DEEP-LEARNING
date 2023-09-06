def dense(inputs, weights):
    return np.matmul(inputs, weights)


input_size = 2
hidden_size = 3
output_size = 1


def initialize_weights():
 # weights for hidden layer, shape: 2x3
    w1 = np.random.uniform(size=(input_size, hidden_size))
 # weights for output layer, shape: 3x1
    w2 = np.random.uniform(size=(hidden_size, output_size))
    return w1, w2


w1, w2 = initialize_weights()


def forward_pass(X):
 # Step 1: Calculate weighted average of inputs (output shape: 4x3)
    net_hidden = dense(X, w1)

 # Step 2: Calculate the result of the sigmoid activation function (shape: 4x3)
    act_hidden = sigmoid(net_hidden)

 # Step 3: Calculate output of neural network (output shape: 4x1)
    y_hat = dense(act_hidden, w2)

    return act_hidden, y_hat


def mse(y_hat, y):
    residual = y_hat - y
    error = np.mean(0.5 * (residual ** 2))
    return residual, error


def backward(X, y_hat, act_hidden):
 # Step 1: Calculate error
    residual, error = mse(y_hat, y)

 # Step 2: calculate gradient wrt w2
    N = X.shape[0]
    dL_dy = 1.0 / N * residual  # shape (4, 1)
    dy_dw2 = act_hidden  # shape (4, 3)
    dL_dw2 = np.matmul(dL_dy.T, dy_dw2)  # shape (1, 3)

 # According to the math, `dL_dw2` is a row-vector, however, `w2` is a column-vector.
 # To prevent erroneous numpy broadcasting during the gradient update, we must make
 # sure that `dL_dw2` is also a column-vector.
    dL_dw2 = dL_dw2.T

 # Step 3: calculate gradient wrt w1
    da_dh = sigmoid_(act_hidden)
    dL_dw1 = np.zeros_like(w1)
    for i in range(w1.shape[0]):
        for j in range(w1.shape[1]):
         # Note: using `residual[:, 0]` instead of just `residual` is important here, as otherwise
         # numpy broadcasting will make `s` a 4x4 matrix, which is wrong
            s = residual[:, 0] * w2[j, 0] * da_dh[:, j] * X[:, i]
            dL_dw1[i, j] = np.mean(s)
    return dL_dw2, dL_dw


def d_faster(X, y_hat, act_hidden):
 # Step 1: Calculate error
    residual, error = mse(y_hat, y)

 # Step 2: calculate gradient wrt w2
    N = X.shape[0]
    dL_dy = 1.0 / N * residual  # shape (4, 1)
    dy_dw2 = act_hidden  # shape (4, 3)
    dL_dw2 = np.matmul(dL_dy.T, dy_dw2)  # shape (1, 3)

 # According to the math, `dL_dw2` is a row-vector, however, `w2` is a column-vector.
 # To prevent erroneous numpy broadcasting during the gradient update, we must make
 # sure that `dL_dw2` is also a column-vector.
    dL_dw2 = dL_dw2.T

 # Step 3: calculate gradient wrt w1
    da_dh = sigmoid_(act_hidden)  # shape (4, 3)asting by numpy
    dL_dw1 = 1.0 / N * np.matmul(X.T, dL_dw1)  # shape (2, 3)

    return dL_dw2, dL_dw1


def backward_pass(X, y_hat, act_hidden):
 # Step 1: Calculate error
    residual, error = mse(y_hat, y)

 # Step 2: calculate gradient wrt w2
    N = X.shape[0]
    dL_dy = 1.0 / N * residual  # shape (4, 1)
    dy_dw2 = act_hidden  # shape (4, 3)
    dL_dw2 = np.matmul(dL_dy.T, dy_dw2)  # shape (1, 3)

 # According to the math, `dL_dw2` is a row-vector, however, `w2` is a column-vector.
 # To prevent erroneous numpy broadcasting during the gradient update, we must make
 # sure that `dL_dw2` is also a column-vector.
    dL_dw2 = dL_dw2.T

 # Step 3: calculate gradient wrt w1
    dL_dw1 = 1.0 / N * \
        np.matmul(X.T, np.matmul(residual, w2.T) * sigmoid_(act_hidden))

    return dL_dw2, dL_dw1, error


n_epochs = 10000
learning_rate = 0.1
training_errors = []
# re-initialize the weights to be sure we start fresh
w1, w2 = initialize_weights()
for epoch in range(n_epochs):
 # Step 1: forward pass
    act_hidden, y_hat = forward_pass(X)

 # Step 2: backward pass
    dw2, dw1, error = backward_pass(X, y_hat, act_hidden)

 # Step 3: apply gradients scaled by learning rate
    w2 = w2 - learning_rate * dw2
    w1 = w1 - learning_rate * dw1

 # Step 4: some book-keeping and print-out
    if epoch % 200 == 0:
        print('Epoch %d> Training error: %f' % (epoch, error))
    training_errors.append([epoch, error])
# Plot training error progression over time
training_errors = np.asarray(training_errors)
plt.plot(training_errors[:, 0], training_errors[:, 1])
plt.xlabel('Epochs')
plt.ylabel('Training Error')

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_hat = [np.round(forward_pass(x)[1]) for x in X]
# Colors corresponding to class predictions y_hat.
colors = ['green' if y_ == 1 else 'blue' for y_ in y_hat]
fig = plt.figure()
fig.set_figwidth(6)
fig.set_figheight(6)
plt.scatter(X[:, 0], X[:, 1], s=100, c=colors)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
resolution = 20
min_x, min_y = 0.0, 0.0
max_x, max_y = 1.0, 1.0
xv, yv = np.meshgrid(np.linspace(min_x, max_x, resolution),
                     np.linspace(min_y, max_y, resolution))
X_extended = np.concatenate(
    [xv[..., np.newaxis], yv[..., np.newaxis]], axis=-1)
X_extended = np.reshape(X_extended, [-1, 2])
y_hat = [np.round(forward_pass(x)[1]) for x in X_extended]
# Colors corresponding to class predictions y_hat.
colors = ['green' if y_ == 1 else 'blue' for y_ in y_hat]
fig = plt.figure()
fig.set_figwidth(6)
fig.set_figheight(6)
plt.scatter(X_extended[:, 0], X_extended[:, 1], s=200, c=colors)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()