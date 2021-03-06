import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, residual_layers_count=20, hidden_state_dim=64, value_fc_dim=128):
        super(Model, self).__init__()

        # Parameters
        self.residual_layers_count = residual_layers_count
        self.hidden_state_dim = hidden_state_dim
        self.value_fc_dim = value_fc_dim

        # Residual layers
        self.residual_conv_1 = nn.Conv2d(in_channels=self.hidden_state_dim,
                                         out_channels=self.hidden_state_dim,
                                         kernel_size=3,
                                         padding=1)

        self.residual_conv_2 = nn.Conv2d(in_channels=self.hidden_state_dim,
                                         out_channels=self.hidden_state_dim,
                                         kernel_size=3,
                                         padding=1)

        # Convolutional layer
        self.convolutional_conv_1 = nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden_state_dim,
                                              kernel_size=3,
                                              padding=1)

        # Policy head
        self.policy_conv_1 = nn.Conv2d(in_channels=self.hidden_state_dim, out_channels=2, kernel_size=1)
        self.policy_batch_norm_1 = nn.BatchNorm2d(num_features=2)
        self.policy_fc1 = nn.Linear(in_features=2 * 6 * 7, out_features=7)  # 7 possible moves for ConnectFour

        # Value head
        self.value_conv_1 = nn.Conv2d(in_channels=self.hidden_state_dim, out_channels=1, kernel_size=1)
        self.value_batch_norm_1 = nn.BatchNorm2d(num_features=1)
        self.value_fc1 = nn.Linear(in_features=1 * 6 * 7, out_features=self.value_fc_dim)
        self.value_fc2 = nn.Linear(in_features=self.value_fc_dim, out_features=1)

        # Helper function
        self.batch_norm = nn.BatchNorm2d(num_features=self.hidden_state_dim)

    def forward(self, x):
        x = self.convolutional_layer(x)

        for _ in range(self.residual_layers_count):
            x = self.residual_layer(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

    def convolutional_layer(self, x):
        x = self.convolutional_conv_1(x)
        x = self.batch_norm(x)
        x = torch.relu(x)

        return x

    def residual_layer(self, input_block):
        x = self.residual_conv_1(input_block)
        x = self.batch_norm(x)
        x = torch.relu(x)

        x = self.residual_conv_2(x)
        x = self.batch_norm(x)

        x = x + input_block  # Skip-connection
        x = torch.relu(x)

        return x

    def policy_head(self, x):
        x = self.policy_conv_1(x)
        x = self.policy_batch_norm_1(x)
        x = torch.relu(x)

        x = x.view(-1, 2 * 6 * 7)
        x = self.policy_fc1(x)

        return x

    def value_head(self, x):
        x = self.value_conv_1(x)
        x = self.value_batch_norm_1(x)
        x = torch.relu(x)

        x = x.view(-1, 1 * 6 * 7)
        x = self.value_fc1(x)
        x = torch.relu(x)

        x = self.value_fc2(x)
        x = torch.tanh(x)

        return x


def train_model(model: nn.Module,
                train_data: torch.Tensor,
                epochs: int = 10,
                batch_size: int = 64,
                learning_rate: float = 0.001,
                print_stats_every: int = 200) -> None:
    # Define two losses: one for the policy head; one for the value head
    policy_criterion = nn.MSELoss()
    value_criterion = nn.MSELoss()

    # Define an optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Train for the specified number of epochs
    for epoch in range(epochs):
        # Keep track of the running loss
        running_loss = 0.0

        # Train over the minibatches
        for minibatch_index, (x, (target_policies, target_values)) in enumerate(train_data):
            # Reset the gradients every batch
            optimizer.zero_grad()

            # Feed forward, backpropagation and optimize
            predicted_policies, predicted_values = model(x)

            policy_loss = policy_criterion(input=predicted_policies, target=target_policies)
            value_loss = value_criterion(input=predicted_values, target=target_values)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print stats at regular intervals to monitor the training
            if not minibatch_index % print_stats_every:
                print(f"[{epoch}, {minibatch_index}] loss: {running_loss / print_stats_every}")

                running_loss = 0.0

    print("Done.")


def main():
    model = Model()

    dummy_state = torch.rand((1, 3, 6, 7))

    with torch.no_grad():
        policy, value = model(dummy_state)

    print("Value", value)
    print("Policy", policy)


if __name__ == "__main__":
    main()