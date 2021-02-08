import torch.utils.data
from torch.utils.data.dataloader import default_collate

from batch import BatchSubstructContext, BatchMasking, BatchAE, BatchContrast, BatchContrast3, BatchContrast2, BatchContrast4, BatchContrast5, \
    BatchContrast6, BatchContrast7, BatchMasSubstruct, BatchSplitLabel, BatchMaskedData, BatchFlowContrastData, BatchFlowContrastDataVTwo, \
    BatchMultiMask, BatchContrastSimBased, BatchContrastSimBasedExpPos, BatchContrastSimBasedExpPosPer, BatchContrastSimBasedExpPosComNeg, \
    BatchContrastSimBasedExpPosComNegMg, BatchContrastList, BatchContrastSubgraph

class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs)

class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)

class DataLoaderContrastSubgraph(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrastSubgraph, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrastSubgraph.from_data_list(data_list),
            **kwargs)

class DataLoaderContrastSimBased(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrastSimBased, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrastSimBased.from_data_list(data_list),
            **kwargs)

class DataLoaderContrastSimBasedMultiPos(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrastSimBasedMultiPos, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrastSimBasedExpPos.from_data_list(data_list),
            **kwargs)

import numpy as np
import os

class DataLoaderContrastSimBasedMultiPosPer(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        processed_path = "./dataset/zinc_standard_agent/processed"
        degree_np_array = np.load(os.path.join(processed_path, "big_graph_degree_array_np_array.npy"))
        self.degree_unnormalized_tensor = torch.from_numpy(degree_np_array)
        self.degree_unnormalized_tensor = self.degree_unnormalized_tensor / torch.sum(self.degree_unnormalized_tensor,
                                                                                      dim=0,
                                                                                      keepdim=False)
        self.dataset = dataset
        super(DataLoaderContrastSimBasedMultiPosPer, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrastSimBasedExpPosPer.from_data_list(data_list,
                                                                                       degree_file=self.degree_unnormalized_tensor,
                                                                                       dataset=self.dataset),
            **kwargs)

class DataLoaderContrastSimBasedMultiPosComNeg(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, args=None, **kwargs):
        if args.env != "jizhi":
            processed_path = "./dataset/zinc_standard_agent/processed"
        else:
            processed_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/zinc_standard_agent/processed"
        # degree_np_array = np.load(os.path.join(processed_path, "big_graph_degree_array_np_array.npy"))
        degree_np_array = np.load(os.path.join(processed_path, "big_graph_degree_array_np_array_no_exp.npy"))

        self.degree_unnormalized_tensor = torch.from_numpy(degree_np_array)
        print(self.degree_unnormalized_tensor.size())
        print("before pow", self.degree_unnormalized_tensor.max())
        self.degree_unnormalized_tensor = torch.pow(self.degree_unnormalized_tensor, 0.75)
        print(self.degree_unnormalized_tensor.size(), self.degree_unnormalized_tensor.max(), "after_pow")
        self.degree_unnormalized_tensor = self.degree_unnormalized_tensor / torch.sum(self.degree_unnormalized_tensor,
                                                                                      dim=0,
                                                                                      keepdim=False)
        self.dataset = dataset
        self.args = args
        super(DataLoaderContrastSimBasedMultiPosComNeg, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrastSimBasedExpPosComNeg.from_data_list(data_list,
                                                                                       degree_file=self.degree_unnormalized_tensor,
                                                                                       dataset=self.dataset,
                                                                                          args=self.args),
            **kwargs)

import torch_geometric
class DataLoaderContrastSimBasedMultiPosComNegPyg(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, args=None, **kwargs):
        if args.env != "jizhi":
            processed_path = "./dataset/zinc_standard_agent/processed"
        else:
            processed_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/zinc_standard_agent/processed"
        # degree_np_array = np.load(os.path.join(processed_path, "big_graph_degree_array_np_array.npy"))
        degree_np_array = np.load(os.path.join(processed_path, "big_graph_degree_array_np_array_no_exp.npy"))

        self.degree_unnormalized_tensor = torch.from_numpy(degree_np_array)
        print(self.degree_unnormalized_tensor.size())
        print("before pow", self.degree_unnormalized_tensor.max())
        self.degree_unnormalized_tensor = torch.pow(self.degree_unnormalized_tensor, 0.75)
        print(self.degree_unnormalized_tensor.size(), self.degree_unnormalized_tensor.max(), "after_pow")
        self.degree_unnormalized_tensor = self.degree_unnormalized_tensor / torch.sum(self.degree_unnormalized_tensor,
                                                                                      dim=0,
                                                                                      keepdim=False)
        self.dataset = dataset
        self.args = args
        super(DataLoaderContrastSimBasedMultiPosComNegPyg, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrastSimBasedExpPosComNegMg.from_data_list(data_list,
                                                                                       degree_file=self.degree_unnormalized_tensor,
                                                                                       dataset=self.dataset,
                                                                                          args=self.args),
            **kwargs)


class DataLoaderContrastList(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, args=None, **kwargs):

        self.dataset = dataset
        self.args = args
        super(DataLoaderContrastList, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrastList.from_data_list(data_list,
                                                                              args=self.args),
            **kwargs)


class DataLoaderTriplet(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderTriplet, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSplitLabel.from_data_list(data_list),
            **kwargs)


class DataLoaderAE(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs)

class DataLoaderContrast(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrast, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrast.from_data_list(data_list),
            **kwargs)

class DataLoaderContrast2(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrast2, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrast2.from_data_list(data_list),
            **kwargs)

class DataLoaderContrast3(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrast3, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrast3.from_data_list(data_list),
            **kwargs)

class DataLoaderContrast4(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrast4, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrast4.from_data_list(data_list),
            **kwargs)

class DataLoaderContrast5(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrast5, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrast5.from_data_list(data_list),
            **kwargs)


class DataLoaderContrast6(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrast6, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrast6.from_data_list(data_list),
            **kwargs)

class DataLoaderContrast7(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrast7, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchContrast7.from_data_list(data_list),
            **kwargs)

class DataLoaderMaskSubstruct(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMaskSubstruct, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasSubstruct.from_data_list(data_list),
            **kwargs)

class DataLoaderContrastPair(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrastPair, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)

class DataLoaderMaskedFlow(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMaskedFlow, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMaskedData.from_data_list(data_list),
            **kwargs)

class DataLoaderContrastFlow(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrastFlow, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchFlowContrastData.from_data_list(data_list),
            **kwargs)

class DataLoaderContrastFlowVTwo(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrastFlowVTwo, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchFlowContrastDataVTwo.from_data_list(data_list),
            **kwargs)

class DataLoaderMultiMask(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMultiMask, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMultiMask.from_data_list(data_list),
            **kwargs)