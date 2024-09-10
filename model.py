import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn as nn

# Define the GCN model for forecasting with a fully connected layer
class GCNForecast(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNForecast, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        batch_size, num_nodes, _ = x.size()
        x = x.view(batch_size * num_nodes, -1)  # Reshape to (batch_size * num_nodes, in_channels)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.view(batch_size, num_nodes, -1)  # Reshape back to (batch_size, num_nodes, out_channels)
        x = self.fc(x)
        return x


def load_data(adj_matrix, feature_matrix, labels):
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)
    edge_index = torch.nonzero(adj_matrix).t().contiguous()
    x = torch.tensor(feature_matrix, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float).view(-1, 815, 1)  # Reshape labels to (batch_size, num_nodes, 1)
    return x, edge_index, y


def train(model, x, edge_index, y, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, x, edge_index, y, criterion):
    model.eval()
    out = model(x, edge_index)
    print(out.shape)
    loss = criterion(out, y)
    return loss.item()