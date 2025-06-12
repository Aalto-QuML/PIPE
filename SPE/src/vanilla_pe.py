from typing import List, Callable

import torch
from torch import nn
from torch_geometric.utils import unbatch
from typing import List, Callable
import networkx as nx
import torch
from torch import nn
from torch_geometric.utils import unbatch, to_dense_adj, degree
import sys
from src.stable_expressive_pe import GINPhi
from src.gin import GIN_PH
from src.mlp import MLP
import torch_geometric.utils as tgu
sys.path.append('../../')
from RePHINE.layers.rephine_layer import RephineLayer
from torch_geometric.transforms import AddRandomWalkPE


class NoPE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList

    def __init__(self, pe_dim: int) -> None:
        super().__init__()
        self.out_dim = pe_dim

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        :param Lambda: Eigenvalue vectors. [B, D_pe]
        :param V: Concatenated eigenvector matrices. [N_sum, D_pe]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        # for sanity check
        return 0 * V   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.out_dim

class IdPE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList

    def __init__(self, pe_dim: int) -> None:
        super().__init__()
        self.out_dim = pe_dim

    def forward(
        self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        :param Lambda: Eigenvalue vectors. [B, D_pe]
        :param V: Concatenated eigenvector matrices. [N_sum, D_pe]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        return V   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.out_dim


class RWPE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList

    def __init__(self, pe_dim: int) -> None:
        super().__init__()
        self.pos_enc_dim = 20
        self.out_dim = pe_dim
        self.num_filtrations = 8
        self.out_ph = 64
        self.fil_hid = 16

        self.topo = RephineLayer(
                n_features=20,
                n_filtrations=self.num_filtrations,
                filtration_hidden=self.fil_hid,
                out_dim=self.out_ph,
                diagram_type='rephine',
                dim1=True,
                sig_filtrations=True,
            )
    
    def forward(
        self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,data
    ) -> torch.Tensor:
        """
        :param Lambda: Eigenvalue vectors. [B, D_pe]
        :param V: Concatenated eigenvector matrices. [N_sum, D_pe]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        
        PE = AddRandomWalkPE(walk_length=self.pos_enc_dim)(data).random_walk_pe
        ph_vector = self.topo(PE,data)       
        return PE,ph_vector   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.out_dim
