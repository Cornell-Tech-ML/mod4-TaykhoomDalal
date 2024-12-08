"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        self.layer1 = Linear(2, hidden_layers)

        self.layer2 = Linear(hidden_layers, hidden_layers)

        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        """Forward pass of the network. """
        middle = self.layer1.forward(x).relu()
        end = self.layer2.forward(middle).relu()
        return self.layer3.forward(end).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)

    def forward(self, inputs):
        """Compute y = Wx + b."""
        # Note: W shape is (in_size, out_size)
        #       x shape is (N, in_size)
        #       b shape is (out_size)

        W = self.weights.value.view(1, *self.weights.value.shape)  # (1, in_size, out_size)
        x = inputs.view(*inputs.shape, 1)  # (N, in_size, 1)
        b = self.bias.value.view(1, *self.bias.value.shape) # (1, out_size)

        multiplied = W * x  # (N, in_size, out_size)
        Wx = multiplied.sum(1)  # (N, 1, out_size) (summation does not remove the singleton dimension)

        Wx = Wx.view(Wx.shape[0], Wx.shape[2])  # (N, out_size) (remove singleton dimension)

        y = Wx + b  # (N, out_size)

        return y

def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
