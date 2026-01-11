"""
BrainNetCNN reproduction (Kawahara et al., 2017) — corrected E2E / E2N / N2G blocks.

Notes:
- Keeps the original class and parameter names you provided (E2EBlock, E2NBlock, N2GBlock, BrainNetCNN).
- E2EBlock internally builds two 1-D convolutions (1 x N and N x 1) and sums their outputs,
  matching the "edge-to-edge" filter construction described in the paper / implementations.
- E2N reduces edges -> nodes using a 1-D conv; N2G collapses nodes -> graph using a conv across nodes.
- Uses BatchNorm and ReLU activations (standard reproductions use ReLU). If you prefer LeakyReLU
  with the original negative_slope you had, I can switch that while keeping the same names.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class E2EBlock(nn.Module):
    """Edge-to-Edge (E2E) block implemented as the sum of a (1 x N) and (N x 1) 1-D filters.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        num_nodes (int): number of nodes (N) to set kernel sizes, normally is the width/height of the input adjacency matrix
    """
    def __init__(self, in_channels, out_channels, num_nodes):
        super(E2EBlock, self).__init__()

        # conv_row: kernel (1, num_nodes)
        self.conv_row = nn.Conv2d(in_channels, out_channels, kernel_size=(1, num_nodes), bias=True)
        # conv_col: kernel (num_nodes, 1)
        self.conv_col = nn.Conv2d(in_channels, out_channels, kernel_size=(num_nodes, 1), bias=True)

        self.bn = nn.BatchNorm2d(out_channels)

        # store num_nodes for shape checks in forward
        self.num_nodes = num_nodes

    def forward(self, x):
        # x shape: (B, C_in, N, N)
        B, C, N1, N2 = x.shape
        if N1 != self.num_nodes or N2 != self.num_nodes:
            # allow flexible shapes but warn — kernels were constructed for a particular num_nodes
            # If shapes differ, raise a helpful message.
            raise RuntimeError(f"E2EBlock expected input with spatial size {self.num_nodes}x{self.num_nodes}, "
                               f"but got {N1}x{N2}.")

        # conv_row produces shape (B, out, N, 1) — because kernel width == N
        out_row = self.conv_row(x)   # -> (B, out_channels, N, 1)
        # conv_col produces (B, out, 1, N)
        out_col = self.conv_col(x)   # -> (B, out_channels, 1, N)

        # To sum them and produce an (N x N) response, we broadcast each to (N x N)
        # Expand out_row across the width and out_col across the height:
        out_row_exp = out_row.expand(-1, -1, self.num_nodes, -1)  # (B, out, N, N)
        out_col_exp = out_col.expand(-1, -1, -1, self.num_nodes)  # (B, out, N, N)

        out = out_row_exp + out_col_exp
        out = self.bn(out)
        out = F.leaky_relu(out, negative_slope=0.33, inplace=True)
        return out


class E2NBlock(nn.Module):
    """Edge-to-Node block: collapse edge matrix into node features.

    Signature kept: (in_channels, out_channels, kernel_size).
    Typical kernel_size used: (num_nodes, 1) or (1, num_nodes) depending on orientation.
    We implement the conv using the provided kernel_size and then squeeze the reduced dimension so output
    becomes (B, out_channels, 1, N) or (B, out_channels, N, 1) which is then transposed/handled by downstream.
    """
    def __init__(self, in_channels, out_channels, num_nodes):
        super(E2NBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(num_nodes, 1), bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.num_nodes = num_nodes
        self.kernel = (num_nodes, 1)

    def forward(self, x):
        # x: (B, C_in, N, N)
        B, C, N1, N2 = x.shape
        out = self.conv(x)  # shape depends on kernel (typically (B, out, 1, N) or (B, out, N,1))
        out = self.bn(out)
        out = F.leaky_relu(out, negative_slope=0.33, inplace=True)
        return out


class N2GBlock(nn.Module):
    """Node-to-Graph block: collapse node features to graph-level representation.

    Signature changed to include num_nodes to build a conv of kernel (1, num_nodes) that collapses node-dimension.
    Input expected: (B, C_in, 1, N) or (B, C_in, N, 1) — but we'll accept either and apply a conv that collapses the
    node dimension to 1x1. Output is flattened to (B, out_channels).
    """
    def __init__(self, in_channels, out_channels, num_nodes):
        super(N2GBlock, self).__init__()
        # conv with kernel (1, num_nodes) to collapse node dimension (width)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, num_nodes), bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.num_nodes = num_nodes
        self.kernel = (1, num_nodes)

    def forward(self, x):
        # x expected shape: (B, C_in, 1, N) or (B, C_in, N, 1)
        B, C, H, W = x.shape

        # if nodes are in H (i.e., shape N x 1), transpose so node dimension sits on width (1 x N)
        if H == self.num_nodes and W == 1:
            x = x.permute(0, 1, 3, 2)  # (B, C, 1, N)

        # now x should be (B, C, 1, N)
        if x.shape[2] != 1 or x.shape[3] != self.num_nodes:
            raise RuntimeError(f"N2GBlock expected input with shape (B,C,1,{self.num_nodes}) but got {x.shape}")

        out = self.conv(x)  # -> (B, out_channels, 1, 1)
        out = self.bn(out)
        out = F.leaky_relu(out, negative_slope=0.33, inplace=True)
        out = out.view(B, -1)  # flatten to (B, out_channels)
        return out


class BrainNetCNN(nn.Module):
    """
    BrainNetCNN: keeps parameter names similar to your original code but fixes E2E/E2N/N2G
    to match the original design more closely.

    Args:
        num_nodes (int): number of nodes (N)
        num_features (int): number of input channels (usually 1)
        num_classes (int): output classes
        dropout (float): dropout before FC layers
    """
    def __init__(self, num_nodes=90, num_features=1, num_classes=2, dropout=0.5):
        super(BrainNetCNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        
        # E2E layers: use the E2EBlock that builds both 1xN and Nx1 convs internally.
        # Keep your original channel counts but now blocks are faithful to paper.
        self.e2e_1 = E2EBlock(num_features, 32, num_nodes) # first E2E
        self.e2e_2 = E2EBlock(32, 64, num_nodes)  # second E2E (paper allows multiple E2E layers)

        # E2N: collapse edge -> node (kernel usually (num_nodes, 1) or (1, num_nodes))
        # We'll use (num_nodes, 1) so that output is (B, out, 1, N) after conv if input was (B,C,N,N).
        self.e2n = E2NBlock(64, 128, num_nodes)   # TODO: may need to change 128 to 1, matching the torch implementation

        # N2G: collapse node -> graph. Provide num_nodes explicitly.
        self.n2g = N2GBlock(128, 256, num_nodes)

        # Fully connected layers (kept similar to original pattern; adjust as required)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (B, num_features, N, N)
        returns logits (B, num_classes)
        """
        # E2E stack
        x = self.e2e_1(x)   # (B, 32, N, N)
        x = self.e2e_2(x)   # (B, 64, N, N)

        # E2N: collapse edges to node features -> typically (B, 128, 1, N)
        x = self.e2n(x)

        # N2G: collapse node features to graph-level -> (B, 256)
        x = self.n2g(x)

        # FC
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)

        x = self.fc_out(x)  # logits
        return x

    def get_num_params(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
