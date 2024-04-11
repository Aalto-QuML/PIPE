from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedShuffleSplit
import os.path as osp
from sklearn.model_selection import KFold, ShuffleSplit
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
import gudhi as gd
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, to_networkx
from torch_geometric.data import DataLoader
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch_geometric.data import Data
import sys
sys.path.append('../')
from RePHINE.experiment import load_data
import gudhi.representations as tda
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from utils.utils import _forman_curvature_unweighted,pyg_to_simplex_tree

from utils.ph_utils import create_landscapes_gudhi
import networkx as nx

class FilterConstant(object):
  def __init__(self, dim):
    self.dim = dim

  def __call__(self, data):
    data.x = torch.ones(data.num_nodes, self.dim)
    return data


def load_data_perslay(dataset, path_dataset="", verbose=False):
  path_dataset = "./datasets/data_perslay/" + dataset + "/" if not len(path_dataset) else path_dataset
  feat = pd.read_csv(path_dataset + dataset + ".csv", index_col=0, header=0)
  F = np.array(feat)[:, 1:]
  L = np.array(LabelEncoder().fit_transform(np.array(feat["label"])))
  L = OneHotEncoder(sparse=False, categories="auto").fit_transform(L[:, np.newaxis])

  if verbose:
    print("Dataset:", dataset)
    print("Number of observations:", L.shape[0])
    print("Number of classes:", L.shape[1])

  return F, L


def get_tudataset(name, rwr=False, cleaned=False):
  transform = None
  if rwr:
    transform = None
  path = osp.join(osp.dirname(osp.realpath(__file__)), ('rwr' if rwr else ''))
  dataset = TUDataset(path, name, pre_transform=transform, use_edge_attr=rwr, cleaned=cleaned)

  if not hasattr(dataset, 'x'):
    max_degree = 0
    degs = []
    for data in dataset:
      degs += [degree(data.edge_index[0], dtype=torch.long)]
      max_degree = max(max_degree, degs[-1].max().item())
    dataset.transform = FilterConstant(10)  # T.OneHotDegree(max_degree)
  return dataset


def data_split(dataset, seed=42):
  skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
  train_idx, val_test_idx = list(skf_train.split(torch.zeros(len(dataset)), dataset.y))[0]
  skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
  val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), dataset.y[val_test_idx]))[0]
  train_data = dataset[train_idx]
  val_data = dataset[val_test_idx[val_idx]]
  test_data = dataset[val_test_idx[test_idx]]
  return train_data, val_data, test_data

def add_attributes(dataset, features):
  data_list = []
  for i, data in enumerate(dataset):
    data.graph_features = features[i, :].unsqueeze(0)
    data_list.append(data)
  dataset.data, dataset.slices = dataset.collate(data_list)
  return dataset

def get_data(name, perslay_feats=False, seed=42):
  if name == 'ZINC':
    train_data, val_data, test_data = get_zinc()
    num_classes = 1
  elif name == 'ogbg-molhiv':
    train_data, val_data, test_data = get_molhiv()
    num_classes = 2
  else:
    data = get_tudataset(name)
    if perslay_feats:
      features, L = load_data_perslay(name)
      assert (torch.tensor(L).long().argmax(dim=1) - data.y).sum() == 0
      data = add_attributes(data, torch.tensor(features).float())
    num_classes = data.num_classes
    train_data, val_data, test_data = data_split(data, seed=seed)

  stats = dict()
  stats['num_features'] = train_data.num_node_features
  stats['num_classes'] = num_classes


  new_train = []
  new_val = []
  new_test = []

  print(test_data)

  for idx,sample in enumerate(train_data):
    #cycles = nx.cycle_basis(to_networkx(sample,to_undirected=True))
    #print(idx)
    if idx in [888,889,890,887,868,849,841,840,813,752,783,762,719,711,704,690,658,618,609,583,535,536,463,462,461,457,456,445,438,434,425,424,420,402,403,399,350,395,386,378,356,344,323,333,11,13,22,65,83,130,137,138,162,178,272,297,311] : continue
    ph_feats = create_landscapes_gudhi(_forman_curvature_unweighted, to_networkx(sample,to_undirected=True), hom_deg=2, exact=True)
    temp_data = Data(x=sample.x,edge_index=sample.edge_index,y=sample.y,num_nodes=sample.num_nodes,ph_feats=np.array(ph_feats))
    new_train.append(temp_data)
    #sample.ph_feats = np.array(ph_feats)

  for idx,sample in enumerate(val_data):
    #cycles = nx.cycle_basis(to_networkx(sample,to_undirected=True))
    #print(idx)
    if idx in [62,94,99]: continue
    ph_feats = create_landscapes_gudhi(_forman_curvature_unweighted, to_networkx(sample,to_undirected=True), hom_deg=2, exact=True)
    temp_data = Data(x=sample.x,edge_index=sample.edge_index,y=sample.y,num_nodes=sample.num_nodes,ph_feats=np.array(ph_feats))
    new_val.append(temp_data)


  for idx,sample in enumerate(test_data):
    #cycles = nx.cycle_basis(to_networkx(sample,to_undirected=True))
    #print(idx)
    if idx in [31,34,59,62,66,100,101]: continue
    
    ph_feats = create_landscapes_gudhi(_forman_curvature_unweighted, to_networkx(sample,to_undirected=True), hom_deg=2, exact=True)
    temp_data = Data(x=sample.x,edge_index=sample.edge_index,y=sample.y,num_nodes=sample.num_nodes,ph_feats=np.array(ph_feats))
    new_test.append(temp_data)


  #print(ph_feats)
  #breakpoint()
  return new_train, new_val, new_test, stats


def get_data_perslay(dataset, seed=42):
  diags_dict, F, L = load_data(dataset, path_dataset=f"./datasets/data_perslay/{dataset}/")
  F = torch.tensor(F).float()
  L = torch.tensor(L).float()
  thresh = 500
  tmp = Pipeline([
    ("Selector", tda.DiagramSelector(use=True, point_type="finite")),
    ("ProminentPts", tda.ProminentPoints(use=True, num_pts=thresh)),
    ("Scaler", tda.DiagramScaler(use=True, scalers=[([0, 1], MinMaxScaler())])),
    ("Padding", tda.Padding(use=True)),
  ])
  prm = {filt: {"ProminentPts__num_pts": min(thresh, max([len(dgm) for dgm in diags_dict[filt]]))}
         for filt in diags_dict.keys() if max([len(dgm) for dgm in diags_dict[filt]]) > 0}

  # Apply the previous pipeline on the different filtrations.
  diags = []
  for dt in prm.keys():
    param = prm[dt]
    tmp.set_params(**param)
    diags.append(tmp.fit_transform(diags_dict[dt]))

  # For each filtration, concatenate all diagrams in a single array.
  D, npts = [], len(diags[0])
  for dt in range(len(prm.keys())):
    D.append(np.array(np.concatenate([diags[dt][i][np.newaxis,:] for i in range(npts)],axis=0),dtype=np.float32))

  num_pts, num_labels, num_features, num_filt = L.shape[0], L.shape[1], F.shape[1], len(D)


  skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
  train_idx, val_test_idx = list(skf_train.split(torch.zeros(num_pts), L.argmax(dim=1)))[0]

  skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
  val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), L[val_test_idx, :].argmax(dim=1)))[0]

  train_data = [torch.tensor(D[dt][train_idx, :]).float() for dt in range(num_filt)]
  train_features = F[train_idx, :]
  train_labels = L[train_idx, :]

  val_data = [torch.tensor(D[dt][val_test_idx[val_idx], :]).float() for dt in range(num_filt)]
  val_features = F[val_test_idx[val_idx], :]
  val_labels = L[val_test_idx[val_idx], :]

  test_data = [torch.tensor(D[dt][val_test_idx[test_idx], :]).float() for dt in range(num_filt)]
  test_features = F[val_test_idx[test_idx], :]
  test_labels = L[val_test_idx[test_idx], :]

  return train_data, train_features, train_labels, val_data, val_features, val_labels, test_data, test_features, test_labels

def get_zinc():
  path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'ZINC')
  train_data = ZINC(path, subset=True, split='train')
  data_val = ZINC(path, subset=True, split='val')
  data_test = ZINC(path, subset=True, split='test')

  return train_data, data_val, data_test

def get_molhiv():
  path = osp.dirname(osp.realpath(__file__))
  dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=path)
  split_idx = dataset.get_idx_split()
  return dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]

