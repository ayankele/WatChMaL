# generic imports
import numpy as np

# torch imports
import torch

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset, H5Dataset_bonsai

# pyg imports
import torch_geometric.data as PyGData

# torch cluster imports
from torch_cluster import knn_graph


class GnnDataset(H5Dataset):
    def __init__(self, h5file, geometry_file, k_neighbors, transforms=None, is_distributed=True, use_memmap=True):
        """
        Args:
            h5file              ... path to h5 dataset file
            geometry_file       ... path to the geometry file
            k_neighbors         ... number of nearst neighbors used to connect the graph
            transforms          ... transforms to apply
            use_memmap          ... use a memmap and load data into memory as needed (default), otherwise load entire dataset at initialisation
        """
        super().__init__(h5file, use_memmap)

        geo_file = np.load(geometry_file, 'r')
        self.geo_positions = geo_file['position'].astype(np.float32)
        self.geo_orientations = geo_file['orientation'].astype(np.float32)

        self.k_neighbors = k_neighbors

    def __getitem__(self, item):
        super().__getitem__(item)

        hit_positions = self.geo_positions[self.event_hit_pmts - 1, :] #SK cable numbers start at 1
        hit_orientations = self.geo_orientations[self.event_hit_pmts - 1, :]

        n_hits = self.event_hit_pmts.shape[0]

        # define the training feature matrix (x,y,z, e_x, e_y, e_z, charge, time)
        data = np.zeros((6, n_hits))
        data[:3, :n_hits] = hit_positions[:n_hits].T
        data[3:6, :n_hits] = hit_orientations[:n_hits].T
        #data[-2, :n_hits] = self.event_hit_charges[:n_hits]
        #data[-1, :n_hits] = self.event_hit_times[:n_hits]

        data = data.T
        # scale the training feature to be almost of the scale
        #scale = np.array([100., 100., 100., 1., 1., 1., 1., 1000.])
        #data /= scale

        x = torch.tensor(data, dtype=torch.float32)
        # label tensor
        y = torch.tensor(self.labels[item], dtype=torch.int64)
        # graph connecvtivety by k_nn algorithm
        knn_graph_edge_index = knn_graph(x[:, 0:3], k=self.k_neighbors)

        return {"data": PyGData.Data(x=x, y=y, edge_index=knn_graph_edge_index), "labels": y}

class GnnDataset_bonsai(H5Dataset_bonsai):
    def __init__(self, h5file, geometry_file, transforms=None, is_distributed=True, use_memmap=True):
        """
        Args:
            h5file              ... path to h5 dataset file
            geometry_file       ... path to the geometry file
            k_neighbors         ... number of nearst neighbors used to connect the graph
            transforms          ... transforms to apply
            use_memmap          ... use a memmap and load data into memory as needed (default), otherwise load entire dataset at initialisation
        """
        super().__init__(h5file, use_memmap)

        geo_file = np.load(geometry_file, 'r')
        self.geo_positions = geo_file['position'].astype(np.float32)
        self.geo_orientations = geo_file['orientation'].astype(np.float32)

    def __getitem__(self, item):
        super().__getitem__(item)

        hit_positions = self.geo_positions[self.event_hit_pmts - 1, :] #SK cable numbers start at 1
        hit_orientations = self.geo_orientations[self.event_hit_pmts - 1, :]

        n_hits = self.event_hit_pmts.shape[0]

        # define the training feature matrix (x,y,z, e_x, e_y, e_z, charge, time)
        data = np.zeros((n_hits, 10))
        data[:n_hits, :3] = hit_positions[:n_hits]
        data[:n_hits, 3:6] = hit_orientations[:n_hits]
        #data[-2, :n_hits] = self.event_hit_charges[:n_hits]
        #data[-1, :n_hits] = self.event_hit_times[:n_hits]

        data_tvtx = self.tvtx_data[item].reshape((9,10))
        data_tvtx[:,3:] = data_tvtx[:,[5,6,7,3,4,8,9]]

        data = np.concatenate((data_tvtx, data))

        #data = data.T

        # scale the training feature to be almost of the scale
        #scale = np.array([100., 100., 100., 1., 1., 1., 1., 1000.])
        #data /= scale

        x = torch.tensor(data, dtype=torch.float32)
        # label tensor
        y = torch.tensor(self.labels[item], dtype=torch.int64)
        edges = torch.LongTensor(np.copy(self.event_edges.T))
        edge_vars = torch.tensor(self.event_edge_vars.reshape((-1,1)))


        return {"data": PyGData.Data(x=x, y=y, edge_index=edges, edge_attr=edge_vars), "labels": y}
        #return {"data": PyGData.Data(x=x, y=y, edge_index=edges), "labels": y}
