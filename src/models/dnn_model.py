import torch.nn as nn


class DNNModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout,
        activation_function,
    ):
        super(DNNModel, self).__init__()
        layers = []
        current_input_size = input_size

        # Select the appropriate activation function
        if activation_function == "relu":
            activation = nn.ReLU()
        elif activation_function == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_function == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

        for i in range(num_layers):
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(activation)  # Apply the selected activation function
            layers.append(nn.Dropout(dropout))  # Apply dropout
            current_input_size = hidden_size

        layers.append(nn.Linear(current_input_size, output_size))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
