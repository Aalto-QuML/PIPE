from typing import Callable

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from src.mlp import MLP
import sys
sys.path.append('../../')
from RePHINE.layers.rephine_layer import RephineLayer

class GIN(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP],
            bn: bool = False, residual: bool = False
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        self.residual = residual
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINLayer(create_mlp(in_dims, hidden_dims))
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))

        layer = GINLayer(create_mlp(hidden_dims, out_dims))
        self.layers.append(layer)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        for i, layer in enumerate(self.layers):
            X0 = X
            X = layer(X, edge_index)   # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if X.ndim == 3:
                    X = self.batch_norms[i](X.transpose(2, 1)).transpose(2, 1)
                else:
                    X = self.batch_norms[i](X)
            if self.residual:
                X = X + X0
        return X                       # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims


class GIN_PH(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP],
            bn: bool = False, residual: bool = False
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        self.residual = residual
        self.num_filtrations = 8
        self.out_ph = 64
        self.fil_hid = 16

        topo_layers = []

        for i in range(n_layers-1):
            topo = RephineLayer(
                n_features=hidden_dims,
                n_filtrations=self.num_filtrations,
                filtration_hidden=self.fil_hid,
                out_dim=self.out_ph,
                diagram_type='standard',
                dim1=True,
                sig_filtrations=True,
            )
            topo_layers.append(topo)


        self.ph_layers = nn.ModuleList(topo_layers)
        self.ph_pooling_type = "mean"

        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINLayer(create_mlp(in_dims, hidden_dims))
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))

        layer = GINLayer(create_mlp(hidden_dims, out_dims))
        self.layers.append(layer)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, data,mask) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        ph_vectors = []
        for i, layer in enumerate(self.layers):
            #breakpoint
            X0 = X
            X = layer(X, edge_index)
            #breakpoint()
            X_ph = (X*mask.unsqueeze(-1)).sum(dim=1) 
            if i != len(self.layers)-1: ph_vectors += [self.ph_layers[i](X_ph, data)]   # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if X.ndim == 3:
                    X = self.batch_norms[i](X.transpose(2, 1)).transpose(2, 1)
                else:
                    X = self.batch_norms[i](X)
            if self.residual:
                X = X + X0

        ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        return X,ph_embedding                      # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims
    

class GINLayer(MessagePassing):
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, mlp: MLP) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, ***, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True)
        self.mlp = mlp

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        # Contains sum(j in N(i)) {message(j -> i)} for each node i.
        S = self.propagate(edge_index, X=X)   # [N_sum, *** D_in]

        Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
        Z = Z + S                # [N_sum, ***, D_in]
        return self.mlp(Z)       # [N_sum, ***, D_out]

    def message(self, X_j: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, ***, D_in]
        :return: The messages X_j for each edge (j -> i). [E_sum, ***, D_in]
        """
        return X_j   # [E_sum, ***, D_in]

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims
