import torch
from torch_geometric.data import Data, Batch

class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch
        # self.atom_vocab = atom_vocab

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        coar = False
        keys = [set(data.keys) for data in data_list]
        batch_num_nodes = sum([data.num_nodes for data in data_list])
        keys = list(set.union(*keys))
        assert 'batch' not in keys
        if "coarsed_x" in keys:
            coar = True
        # print("coar is", coar)

        batch = BatchMasking()
        # batch.batch_node_assign = []
        # batch_keys = ["batch_node_assign"]
        for key in keys:
            batch[key] = []
        batch.batch = []

        if coar:
            batch.batch_coarsed = []

        cumsum_node = 0
        cumsum_edge = 0
        cumsum_node_coar = 0

        # batch.mask = []
        batch.num_moleculor = len(data_list)
        batch.moleculor_nodes = [0]
        batch.moleculor_edges = [0]
        # alli = {}
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))

            if coar:
                batch.batch_coarsed.append(torch.full((data.coarsed_x.size(0), ), i, dtype=torch.long))

            # batch.batch_node_assign.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices', "masked_atom_indices_a", "masked_atom_indices_b",
                           "part_a_nodes_indices", "part_b_nodes_indices", "edge_droped_index", "mask_indices_a",
                           "mask_indices_b", "sample_unconnected_indices", "sample_connected_indices", "masked_nodes",
                           "edge_index_a", "edge_index_b"]:
                    item = item + cumsum_node
                elif key  in ['connected_edge_indices', "connected_edge_indices_a", "connected_edge_indices_b"]:
                    item = item + cumsum_edge
                elif key in ["coarsed_edge_index"]:
                    item = item + cumsum_node_coar
                batch[key].append(item)

            # data.mask = torch.zeros((num_nodes, batch_num_nodes))
            # node_tensor = torch.arange(num_nodes) + cumsum_node
            # # print(i, num_nodes, cumsum_node)
            # # print("mask_shape1 ", data.mask.size())
            # data.mask[:, node_tensor] = 1.
            # # print("mask_shape2 ", data.mask.size())
            # batch.mask.append(data.mask)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

            if coar:
                cumsum_node_coar += data.coarsed_x.size(0)

            batch.moleculor_nodes.append(cumsum_node)
            batch.moleculor_edges.append(cumsum_edge)

        for key in keys:
            if key == "wave_emb":
                batch[key] = torch.cat(batch[key], dim=0)
            else:
                try:
                    batch[key] = torch.cat(
                        batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
                except:
                    try:
                        batch[key] = torch.cat(batch[key], dim=1)
                    except:
                        for j in batch[key]:
                            print(j.size())
                        raise RuntimeError("{}".format(str(key)))
        batch.batch = torch.cat(batch.batch, dim=-1)

        if coar:
            batch.batch_coarsed = torch.cat(batch.batch_coarsed, dim=-1)

        batch.moleculor_edges = torch.tensor(batch.moleculor_edges, dtype=torch.long)
        batch.moleculor_nodes = torch.tensor(batch.moleculor_nodes, dtype=torch.long)

        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchMultiMask(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMultiMask, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        coar = False

        assert "mask_nodes_idx_list" in data_list[0].keys
        keys = ["x", "edge_index", "edge_attr"]
        # keys = [set(data.keys) for data in data_list]
        batch_num_nodes = sum([data.num_nodes for data in data_list])
        # keys = list(set.union(*keys))
        assert 'batch' not in keys
        # print("coar is", coar)

        batch = BatchMultiMask()
        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        num_mask_version = len(data_list[0].mask_nodes_idx_list)
        batch.masked_nodes_idx = [[] for i in range(num_mask_version)]
        cumcum_masked_nodes = [0 for i in range(num_mask_version)]
        # alli = {}
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))

            for j, mask_nodes_idx in enumerate(data.mask_nodes_idx_list):
                mask_nodes_idx += cumsum_node
                batch.masked_nodes_idx[j].append(mask_nodes_idx)
                cumcum_masked_nodes[j] += mask_nodes_idx.size(0)

            # batch.batch_node_assign.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in keys:
                item = data[key]
                if key in ['edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            # data.mask = torch.zeros((num_nodes, batch_num_nodes))
            # node_tensor = torch.arange(num_nodes) + cumsum_node
            # # print(i, num_nodes, cumsum_node)
            # # print("mask_shape1 ", data.mask.size())
            # data.mask[:, node_tensor] = 1.
            # # print("mask_shape2 ", data.mask.size())
            # batch.mask.append(data.mask)

            cumsum_node += num_nodes

        concated_masked_nodes_idx = list()
        num_masked_nodes = list()
        max_masked_num = max(cumcum_masked_nodes)
        for j in range(num_mask_version):
            tmp_masked_idx = torch.cat(batch.masked_nodes_idx[j], dim=0)
            num_masked_nodes.append(tmp_masked_idx.size(0))
            if (tmp_masked_idx.size(0) < max_masked_num):
                tmp_masked_idx = torch.cat([tmp_masked_idx, torch.full((max_masked_num - tmp_masked_idx.size(0), ), 0,
                                                                       dtype=torch.long)], dim=0)
            concated_masked_nodes_idx.append(tmp_masked_idx.view(1, -1))  #
        batch.masked_nodes_idx = torch.cat(concated_masked_nodes_idx, dim=0)
        batch.num_masked_nodes = torch.tensor(num_masked_nodes, dtype=torch.long)

        for key in keys:
            if key == "wave_emb":
                batch[key] = torch.cat(batch[key], dim=0)
            else:
                try:
                    batch[key] = torch.cat(
                        batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
                except:
                    try:
                        batch[key] = torch.cat(batch[key], dim=1)
                    except:
                        for j in batch[key]:
                            print(j.size())
                        raise RuntimeError("{}".format(str(key)))
        batch.batch = torch.cat(batch.batch, dim=-1)

        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchContrastSimBased(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrastSimBased, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        with_neg = False
        if 'neg_x' in data_list[0]:
            with_neg = True

        batch = BatchContrastSimBased()
        for key in keys:
            batch[key] = []
        batch.batch = []

        #### for positive samples
        num_samples = len(data_list[0].sim_x) # the length of the list --- number of samples
        batch.batch_sim = []
        batch.sim_x = []
        batch.sim_edge_index = []
        batch.sim_edge_attr = []
        cumsum_node_sim = []
        for i in range(num_samples):
            batch.sim_x.append([])
            batch.sim_edge_index.append([])
            batch.sim_edge_attr.append([])
            batch.batch_sim.append([])
            cumsum_node_sim.append(0)

        #### for negative samples
        if with_neg:
            num_neg_samples = len(data_list[0].neg_x)  # the length of the list --- number of samples
            batch.batch_neg = []
            batch.neg_x = []
            batch.neg_edge_index = []
            batch.neg_edge_attr = []
            cumsum_node_neg = 0
        # for i in range(num_neg_samples):
        #     batch.neg_x.append([])
        #     batch.neg_edge_index.append([])
        #     batch.neg_edge_attr.append([])
        #     batch.batch_neg.append([])
        #     cumsum_node_neg.append(0)

        cumsum_node = 0
        cumsum_edge = 0
        # cumsum_node_sim = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes

            for j in range(num_samples):
                num_sim_nodes = data.sim_x[j].size(0)
                batch.batch_sim[j].append(torch.full((num_sim_nodes, ), i, dtype=torch.long))
                data_sim_x = data.sim_x[j]
                data_sim_edge_index = data.sim_edge_index[j]
                data_sim_edge_attr = data.sim_edge_attr[j]
                batch.sim_x[j].append(data_sim_x)
                batch.sim_edge_index[j].append(data_sim_edge_index + cumsum_node_sim[j])
                batch.sim_edge_attr[j].append(data_sim_edge_attr)
                cumsum_node_sim[j] += num_sim_nodes

            if with_neg:
                for j in range(num_neg_samples):
                    num_neg_nodes = data.neg_x[j].size(0)
                    batch.batch_neg.append(torch.full((num_neg_nodes, ), i * num_neg_samples + j, dtype=torch.long))
                    data_neg_x = data.neg_x[j]
                    data_neg_edge_index = data.neg_edge_index[j]
                    data_neg_edge_attr = data.neg_edge_attr[j]
                    batch.neg_x.append(data_neg_x)
                    batch.neg_edge_index.append(data_neg_edge_index + cumsum_node_neg)
                    batch.neg_edge_attr.append(data_neg_edge_attr)

                    cumsum_node_neg += num_neg_nodes

            # num_sim_nodes = data.sim_x.size(0)
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            # batch.batch_sim.append(torch.full((num_sim_nodes, ), i, dtype=torch.long))

            # batch.batch_node_assign.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                if key in ["sim_x", "sim_edge_index", "sim_edge_attr", "neg_x", "neg_edge_index", "neg_edge_attr"]:
                    continue
                item = data[key]
                if key in ['edge_index']:
                    item = item + cumsum_node
                elif key in ['sim_edge_index']:
                    item = item + cumsum_node_sim
                batch[key].append(item)

            cumsum_node += num_nodes
            # cumsum_node_sim += num_sim_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            if key in ["sim_x", "sim_edge_index", "sim_edge_attr", "neg_x", "neg_edge_index", "neg_edge_attr"]:
                continue
            try:
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
            except:
                try:
                    batch[key] = torch.cat(batch[key], dim=1)
                except:
                    for j in batch[key]:
                        print(j.size())
                    raise RuntimeError("{}".format(str(key)))

        for j in range(num_samples):
            batch.sim_x[j] = torch.cat(batch.sim_x[j], dim=0)
            batch.sim_edge_index[j] = torch.cat(batch.sim_edge_index[j], dim=1)
            batch.sim_edge_attr[j] = torch.cat(batch.sim_edge_attr[j], dim=0)
            batch.batch_sim[j] = torch.cat(batch.batch_sim[j], dim=-1)

        if with_neg:
            batch.neg_x = torch.cat(batch.neg_x, dim=0)
            batch.neg_edge_index = torch.cat(batch.neg_edge_index, dim=1)
            batch.neg_edge_attr = torch.cat(batch.neg_edge_attr, dim=0)
            batch.batch_neg = torch.cat(batch.batch_neg, dim=-1)
        batch.batch = torch.cat(batch.batch, dim=-1)
        # batch.batch_sim = torch.cat(batch.batch_sim, dim=-1)

        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchContrastSimBasedExpPosComNeg(Data):  # put one data in one batch and calculate is not available....
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrastSimBasedExpPosComNeg, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list, degree_file=None, dataset=None, args=None):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        # keys = [set(data.keys) for data in data_list]
        assert degree_file is not None
        assert dataset is not None
        # keys = list(set.union(*keys))
        # assert 'batch' not in keys
        assert degree_file.size(0) == 2000000

        num_com_negs = args.num_com_negs

        sampled_neg_idxes = torch.multinomial(degree_file, num_com_negs, replacement=False)
        # print("sampled_neg_idxes", sampled_neg_idxes[: 10])
        neg_node_idx_to_sampled_idx = {int(sampled_neg_idxes[ii]): ii for ii in range(sampled_neg_idxes.size(0))}
        assert len(neg_node_idx_to_sampled_idx) == sampled_neg_idxes.size(0)
        # todo: and data should has its sampled pos node idxes (as a list or a tensor)
        # with_neg = False
        # if 'neg_x' in data_list[0]:
        #     with_neg = True

        #

        batch = BatchContrastSimBasedExpPosComNeg()

        keys = ["x", "edge_index", "edge_attr", "sim_x", "sim_edge_index", "sim_edge_attr",
                "neg_x", "neg_edge_index", "neg_edge_attr"]

        batch.sampled_neg_idxes = sampled_neg_idxes
        # for key in keys:
        #     batch[key] = []
        batch.batch = []  # ori idx to graph idx
        batch.sim_batch = []
        batch.neg_batch = []
        for key in keys:
            batch[key] = []

        neg_gra_idx = 0
        neg_node_cumsum = 0
        sim_gra_idx = 0
        sim_node_cumsum = 0
        gra_idx = 0
        node_cumsum =0

        # get neg data samples
        # print("start to get")
        for ii in range(sampled_neg_idxes.size(0)):
            neg_node_idx = int(sampled_neg_idxes[ii])
            neg_data = dataset.get_data_simple(neg_node_idx)
            batch.neg_batch.append(torch.full((neg_data.x.size(0), ), neg_gra_idx, dtype=torch.long))
            batch["neg_x"].append(neg_data.x)
            batch["neg_edge_index"].append(neg_data.edge_index + neg_node_cumsum)
            batch["neg_edge_attr"].append(neg_data.edge_attr)
            neg_gra_idx += 1
            neg_node_cumsum += neg_data.x.size(0)
        # print("got")
        # print("neg_gra_idx", neg_gra_idx, "neg_node_cumsum", neg_node_cumsum)
        # for positive samples
        num_samples = len(data_list[0].sim_x)
        # print("num_samples", num_samples)

        #### for negative samples
        # if with_neg:
        #     num_neg_samples = len(data_list[0].neg_x)

        # new_data_list = list()
        # batch.x_list = list()
        # batch.edge_index_list = list()
        # batch.edge_attr_list = list()
        # batch.batch_list = list()
        #
        # batch.x_list = torch.empty((0, 2), dtype=torch.long)
        # batch.edge_index_list = torch.empty((2, 0), dtype=torch.long)
        # batch.edge_attr_list = torch.empty((0, 2), dtype=torch.long)
        # batch.batch_list = torch.empty((0,), dtype=torch.long)
        #
        # batch.x_list_slices = list()
        # batch.edge_index_slices = list()
        # batch.edge_attr_slices = list()
        # batch.batch_slices = list()

        # batch_size * num_neg_samples
        batch.masked_pos_idx = torch.full((num_samples, len(data_list)), -1, dtype=torch.long)
        # for each node we need to generate its masked tensor to exclude those negative samples that are already
        # positive samples of those nodes ----
        # if we define negative sampling possibilities by normalized degrees directly, we will never get access to
        # those nodes

        tot_in = 0

        for i, data in enumerate(data_list):
            # num_nodes = data.num_nodes

            # all_x_list = list()
            # all_edge_index_list = list()
            # all_edge_attr_list = list()
            # all_batch_list = list()
            #
            # cumsum_node = 0
            #
            # all_x_list.append(data.x)
            # all_edge_index_list.append(data.edge_index)
            # all_edge_attr_list.append(data.edge_attr)
            #
            # cumsum_node += data.x.size(0)
            # all_batch_list.append(torch.full((data.x.size(0),), 0, dtype=torch.long))

            batch["x"].append(data.x)
            batch["edge_index"].append(data.edge_index + node_cumsum)
            batch["edge_attr"].append(data.edge_attr)
            batch.batch.append(torch.full((data.x.size(0), ), gra_idx, dtype=torch.long))
            gra_idx += 1
            node_cumsum += data.x.size(0)

            for j in range(num_samples):

                num_sim_nodes = data.sim_x[j].size(0)
                # all_batch_list.append(torch.full((num_sim_nodes,), 1 + j, dtype=torch.long))
                data_sim_x = data.sim_x[j]
                data_sim_edge_index = data.sim_edge_index[j]
                data_sim_edge_attr = data.sim_edge_attr[j]
                batch["sim_x"].append(data_sim_x)
                batch["sim_edge_index"].append(data_sim_edge_index + sim_node_cumsum)
                batch["sim_edge_attr"].append(data_sim_edge_attr)
                batch.sim_batch.append(torch.full((num_sim_nodes, ), sim_gra_idx, dtype=torch.long))
                sim_gra_idx += 1
                sim_node_cumsum += num_sim_nodes

                # all_x_list.append(data_sim_x)
                # all_edge_index_list.append(data_sim_edge_index + cumsum_node)
                # all_edge_attr_list.append(data_sim_edge_attr)
                # cumsum_node += num_sim_nodes

                sampled_pos_node_idx = int(data.sim_node_idx[j])
                if sampled_pos_node_idx in neg_node_idx_to_sampled_idx:
                    tot_in += 1
                    sampled_pos_sampled_idx = neg_node_idx_to_sampled_idx[sampled_pos_node_idx]
                    batch.masked_pos_idx[j, i] = sampled_pos_sampled_idx

            # for j in range(num_neg_samples):
            #     num_neg_nodes = data.neg_x[j].size(0)
            #     all_batch_list.append(torch.full((num_neg_nodes,), 1 + num_samples + j, dtype=torch.long))
            #     data_neg_x = data.neg_x[j]
            #     data_neg_edge_index = data.neg_edge_index[j]
            #     data_neg_edge_attr = data.neg_edge_attr[j]
            #     all_x_list.append(data_neg_x)
            #     all_edge_index_list.append(data_neg_edge_index + cumsum_node)
            #     all_edge_attr_list.append(data_neg_edge_attr)
            #
            #     cumsum_node += num_neg_nodes

            # data.all_x = torch.cat(all_x_list, dim=0)
            # data.all_edge_index = torch.cat(all_edge_index_list, dim=1)
            # data.all_edge_attr = torch.cat(all_edge_attr_list, dim=0)
            # data.all_batch = torch.cat(all_batch_list, dim=0)
            # new_data_list.append(data)
            # tmp_x_list = torch.cat(all_x_list, dim=0)
            # tmp_edge_index_list = torch.cat(all_edge_index_list, dim=1)
            # tmp_edge_attr_list = torch.cat(all_edge_attr_list, dim=0)
            # tmp_batch_list = torch.cat(all_batch_list, dim=0)
            #
            # batch.x_list_slices.append((batch.x_list.size(0), batch.x_list.size(0) + tmp_x_list.size(0)))
            # batch.edge_index_slices.append((batch.edge_index_list.size(1), batch.edge_index_list.size(1) + \
            #                                 tmp_edge_index_list.size(1)))
            # batch.edge_attr_slices.append((batch.edge_attr_list.size(0), batch.edge_attr_list.size(0) + \
            #                                tmp_edge_attr_list.size(0)))
            # batch.batch_slices.append((batch.batch_list.size(0), batch.batch_list.size(0) + \
            #                            tmp_batch_list.size(0)))
            #
            # batch.x_list = torch.cat([batch.x_list, tmp_x_list], dim=0)
            # batch.edge_index_list = torch.cat([batch.edge_index_list, tmp_edge_index_list], dim=1)
            # batch.edge_attr_list = torch.cat([batch.edge_attr_list, tmp_edge_attr_list], dim=0)
            # batch.batch_list = torch.cat([batch.batch_list, tmp_batch_list], dim=0)
        # print("gra_idx", gra_idx, "node_cumsum", node_cumsum)
        # print("sim_gra_idx", sim_gra_idx, "sim_node_cumsum", sim_node_cumsum)

        batch.batch = torch.cat(batch.batch, dim=0)
        batch.sim_batch = torch.cat(batch.sim_batch, dim=0)
        batch.neg_batch = torch.cat(batch.neg_batch, dim=0)

        # print(tot_in)
        batch.tot_in = torch.tensor([tot_in], dtype=torch.long)
        for key in keys:
            if key in ["edge_index", "sim_edge_index", "neg_edge_index"]:
                batch[key] = torch.cat(batch[key], dim=1)
            else:
                batch[key] = torch.cat(batch[key], dim=0)

        # batch.x_list_slices = torch.LongTensor(batch.x_list_slices)
        # batch.edge_index_slices = torch.LongTensor(batch.edge_index_slices)
        # batch.edge_attr_slices = torch.LongTensor(batch.edge_attr_slices)
        # batch.batch_slices = torch.LongTensor(batch.batch_slices)
        # print("in batch hhh")
        # batch.data_list = new_data_list
        # print("get data list", len(batch.data_list))
        return batch
        # return batch.contiguous() # don't understand the meaning and function of batchcontiguous();...

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchContrastList(Data):  # put one data in one batch and calculate is not available....
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrastList, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list, args=None):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        # keys = [set(data.keys) for data in data_list]
        # assert degree_file is not None
        # assert dataset is not None
        # keys = list(set.union(*keys))
        # assert 'batch' not in keys
        # assert degree_file.size(0) == 2000000

        batch = BatchContrastList()

        batch.data_list = data_list
        batch.batch = []
        batch.y = list()
        for i, data in enumerate(data_list):
            batch.batch.append(torch.full((data.x.size(0), ), i, dtype=torch.long))
            batch.y.append(data.y)
        batch.batch = torch.cat(batch.batch, dim=0)
        batch.y = torch.cat(batch.y, dim=0)
        return batch


class BatchContrastSimBasedExpPosComNegMg(Data):  # put one data in one batch and calculate is not available....
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrastSimBasedExpPosComNegMg, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list, degree_file=None, dataset=None, args=None):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        # keys = [set(data.keys) for data in data_list]
        assert degree_file is not None
        assert dataset is not None
        # keys = list(set.union(*keys))
        # assert 'batch' not in keys
        assert degree_file.size(0) == 2000000

        num_com_negs = args.num_com_negs

        sampled_neg_idxes = torch.multinomial(degree_file, num_com_negs, replacement=False)
        # print("sampled_neg_idxes", sampled_neg_idxes[: 10])
        neg_node_idx_to_sampled_idx = {int(sampled_neg_idxes[ii]): ii for ii in range(sampled_neg_idxes.size(0))}
        assert len(neg_node_idx_to_sampled_idx) == sampled_neg_idxes.size(0)
        # todo: and data should has its sampled pos node idxes (as a list or a tensor)
        # with_neg = False
        # if 'neg_x' in data_list[0]:
        #     with_neg = True

        #

        batch = BatchContrastSimBasedExpPosComNegMg()

        # keys = ["x", "edge_index", "edge_attr", "sim_x", "sim_edge_index", "sim_edge_attr",
        #         "neg_x", "neg_edge_index", "neg_edge_attr"]

        batch.sampled_neg_idxes = sampled_neg_idxes
        # for key in keys:
        #     batch[key] = []
        batch.batch = []  # ori idx to graph idx
        batch.sim_batch = []
        batch.neg_batch = []
        # for key in keys:
        #     batch[key] = []

        neg_gra_idx = 0
        # neg_node_cumsum = 0
        sim_gra_idx = 0
        # sim_node_cumsum = 0
        gra_idx = 0
        # node_cumsum =0

        # get neg data samples
        # print("start to get")
        batch.neg_data_list = []
        batch.data_list = data_list
        batch.sim_data_list = []

        for ii in range(sampled_neg_idxes.size(0)):
            neg_node_idx = int(sampled_neg_idxes[ii])
            neg_data = dataset.get_data_simple(neg_node_idx)
            batch.neg_batch.append(torch.full((neg_data.x.size(0), ), neg_gra_idx, dtype=torch.long))
            # batch["neg_x"].append(neg_data.x)
            # batch["neg_edge_index"].append(neg_data.edge_index + neg_node_cumsum)
            # batch["neg_edge_attr"].append(neg_data.edge_attr)
            batch.neg_data_list.append(neg_data)
            neg_gra_idx += 1
            # neg_node_cumsum += neg_data.x.size(0)
        # print("got")
        # print("neg_gra_idx", neg_gra_idx, "neg_node_cumsum", neg_node_cumsum)
        # for positive samples
        num_samples = len(data_list[0].sim_list)
        # print("num_samples", num_samples)

        #### for negative samples
        # if with_neg:
        #     num_neg_samples = len(data_list[0].neg_x)

        # new_data_list = list()
        # batch.x_list = list()
        # batch.edge_index_list = list()
        # batch.edge_attr_list = list()
        # batch.batch_list = list()
        #
        # batch.x_list = torch.empty((0, 2), dtype=torch.long)
        # batch.edge_index_list = torch.empty((2, 0), dtype=torch.long)
        # batch.edge_attr_list = torch.empty((0, 2), dtype=torch.long)
        # batch.batch_list = torch.empty((0,), dtype=torch.long)
        #
        # batch.x_list_slices = list()
        # batch.edge_index_slices = list()
        # batch.edge_attr_slices = list()
        # batch.batch_slices = list()

        # batch_size * num_neg_samples
        batch.masked_pos_idx = torch.full((num_samples, len(data_list)), -1, dtype=torch.long)
        # for each node we need to generate its masked tensor to exclude those negative samples that are already
        # positive samples of those nodes ----
        # if we define negative sampling possibilities by normalized degrees directly, we will never get access to
        # those nodes

        tot_in = 0

        for i, data in enumerate(data_list):
            # num_nodes = data.num_nodes

            # all_x_list = list()
            # all_edge_index_list = list()
            # all_edge_attr_list = list()
            # all_batch_list = list()
            #
            # cumsum_node = 0
            #
            # all_x_list.append(data.x)
            # all_edge_index_list.append(data.edge_index)
            # all_edge_attr_list.append(data.edge_attr)
            #
            # cumsum_node += data.x.size(0)
            # all_batch_list.append(torch.full((data.x.size(0),), 0, dtype=torch.long))
            batch.sim_data_list += data.sim_list

            # batch["x"].append(data.x)
            # batch["edge_index"].append(data.edge_index + node_cumsum)
            # batch["edge_attr"].append(data.edge_attr)
            batch.batch.append(torch.full((data.x.size(0), ), gra_idx, dtype=torch.long))
            gra_idx += 1
            # node_cumsum += data.x.size(0)

            for j in range(num_samples):
            #
            #     num_sim_nodes = data.sim_x[j].size(0)
                num_sim_nodes = data.sim_list[j].x.size(0)
            #     # all_batch_list.append(torch.full((num_sim_nodes,), 1 + j, dtype=torch.long))
            #     data_sim_x = data.sim_x[j]
            #     data_sim_edge_index = data.sim_edge_index[j]
            #     data_sim_edge_attr = data.sim_edge_attr[j]
            #     batch["sim_x"].append(data_sim_x)
            #     batch["sim_edge_index"].append(data_sim_edge_index + sim_node_cumsum)
            #     batch["sim_edge_attr"].append(data_sim_edge_attr)
                batch.sim_batch.append(torch.full((num_sim_nodes, ), sim_gra_idx, dtype=torch.long))
                sim_gra_idx += 1
            #     sim_node_cumsum += num_sim_nodes
            #
            #     # all_x_list.append(data_sim_x)
            #     # all_edge_index_list.append(data_sim_edge_index + cumsum_node)
            #     # all_edge_attr_list.append(data_sim_edge_attr)
            #     # cumsum_node += num_sim_nodes
            #
                sampled_pos_node_idx = int(data.sim_node_idx[j])
                if sampled_pos_node_idx in neg_node_idx_to_sampled_idx:
                    tot_in += 1
                    sampled_pos_sampled_idx = neg_node_idx_to_sampled_idx[sampled_pos_node_idx]
                    batch.masked_pos_idx[j, i] = sampled_pos_sampled_idx

            # for j in range(num_neg_samples):
            #     num_neg_nodes = data.neg_x[j].size(0)
            #     all_batch_list.append(torch.full((num_neg_nodes,), 1 + num_samples + j, dtype=torch.long))
            #     data_neg_x = data.neg_x[j]
            #     data_neg_edge_index = data.neg_edge_index[j]
            #     data_neg_edge_attr = data.neg_edge_attr[j]
            #     all_x_list.append(data_neg_x)
            #     all_edge_index_list.append(data_neg_edge_index + cumsum_node)
            #     all_edge_attr_list.append(data_neg_edge_attr)
            #
            #     cumsum_node += num_neg_nodes

            # data.all_x = torch.cat(all_x_list, dim=0)
            # data.all_edge_index = torch.cat(all_edge_index_list, dim=1)
            # data.all_edge_attr = torch.cat(all_edge_attr_list, dim=0)
            # data.all_batch = torch.cat(all_batch_list, dim=0)
            # new_data_list.append(data)
            # tmp_x_list = torch.cat(all_x_list, dim=0)
            # tmp_edge_index_list = torch.cat(all_edge_index_list, dim=1)
            # tmp_edge_attr_list = torch.cat(all_edge_attr_list, dim=0)
            # tmp_batch_list = torch.cat(all_batch_list, dim=0)
            #
            # batch.x_list_slices.append((batch.x_list.size(0), batch.x_list.size(0) + tmp_x_list.size(0)))
            # batch.edge_index_slices.append((batch.edge_index_list.size(1), batch.edge_index_list.size(1) + \
            #                                 tmp_edge_index_list.size(1)))
            # batch.edge_attr_slices.append((batch.edge_attr_list.size(0), batch.edge_attr_list.size(0) + \
            #                                tmp_edge_attr_list.size(0)))
            # batch.batch_slices.append((batch.batch_list.size(0), batch.batch_list.size(0) + \
            #                            tmp_batch_list.size(0)))
            #
            # batch.x_list = torch.cat([batch.x_list, tmp_x_list], dim=0)
            # batch.edge_index_list = torch.cat([batch.edge_index_list, tmp_edge_index_list], dim=1)
            # batch.edge_attr_list = torch.cat([batch.edge_attr_list, tmp_edge_attr_list], dim=0)
            # batch.batch_list = torch.cat([batch.batch_list, tmp_batch_list], dim=0)
        # print("gra_idx", gra_idx, "node_cumsum", node_cumsum)
        # print("sim_gra_idx", sim_gra_idx, "sim_node_cumsum", sim_node_cumsum)

        batch.batch = torch.cat(batch.batch, dim=0)
        batch.sim_batch = torch.cat(batch.sim_batch, dim=0)
        batch.neg_batch = torch.cat(batch.neg_batch, dim=0)
        #
        # # print(tot_in)
        batch.tot_in = torch.tensor([tot_in], dtype=torch.long)
        # for key in keys:
        #     if key in ["edge_index", "sim_edge_index", "neg_edge_index"]:
        #         batch[key] = torch.cat(batch[key], dim=1)
        #     else:
        #         batch[key] = torch.cat(batch[key], dim=0)

        # batch.x_list_slices = torch.LongTensor(batch.x_list_slices)
        # batch.edge_index_slices = torch.LongTensor(batch.edge_index_slices)
        # batch.edge_attr_slices = torch.LongTensor(batch.edge_attr_slices)
        # batch.batch_slices = torch.LongTensor(batch.batch_slices)
        # print("in batch hhh")
        # batch.data_list = new_data_list
        # print("get data list", len(batch.data_list))
        return batch
        # return batch.contiguous() # don't understand the meaning and function of batchcontiguous();...

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1



class BatchContrastSimBasedExpPosPer(Data):   # put one data in one batch and calculate is not available....
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrastSimBasedExpPosPer, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list, degree_file=None):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        assert degree_file is not None
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        sampled_neg_idxes = torch.multinomial(degree_file, 256)
        neg_node_idx_to_sampled_idx = {int(degree_file[ii]): ii for ii in range(sampled_neg_idxes.size(0))}
        # todo: and data should has its sampled pos node idxes (as a list or a tensor)
        with_neg = False
        if 'neg_x' in data_list[0]:
            with_neg = True

        batch = BatchContrastSimBasedExpPosPer()
        for key in keys:
            batch[key] = []
        batch.batch = []

        #### for positive samples
        num_samples = len(data_list[0].sim_x)

        #### for negative samples
        if with_neg:
            num_neg_samples = len(data_list[0].neg_x)

        new_data_list = list()
        batch.x_list = list()
        batch.edge_index_list = list()
        batch.edge_attr_list = list()
        batch.batch_list = list()

        batch.x_list = torch.empty((0, 2), dtype=torch.long)
        batch.edge_index_list = torch.empty((2, 0), dtype=torch.long)
        batch.edge_attr_list = torch.empty((0, 2), dtype=torch.long)
        batch.batch_list = torch.empty((0, ), dtype=torch.long)

        batch.x_list_slices = list()
        batch.edge_index_slices = list()
        batch.edge_attr_slices = list()
        batch.batch_slices = list()

        # batch_size * num_neg_samples
        batch.masked_pos_idx = torch.full((num_samples, 256), -1, dtype=torch.long)

        for i, data in enumerate(data_list):
            # num_nodes = data.num_nodes

            all_x_list = list()
            all_edge_index_list = list()
            all_edge_attr_list = list()
            all_batch_list = list()

            cumsum_node = 0

            all_x_list.append(data.x)
            all_edge_index_list.append(data.edge_index)
            all_edge_attr_list.append(data.edge_attr)

            cumsum_node += data.x.size(0)
            all_batch_list.append(torch.full((data.x.size(0), ), 0, dtype=torch.long))

            for j in range(num_samples):

                num_sim_nodes = data.sim_x[j].size(0)
                all_batch_list.append(torch.full((num_sim_nodes,), 1 + j, dtype=torch.long))
                data_sim_x = data.sim_x[j]
                data_sim_edge_index = data.sim_edge_index[j]
                data_sim_edge_attr = data.sim_edge_attr[j]
                all_x_list.append(data_sim_x)
                all_edge_index_list.append(data_sim_edge_index + cumsum_node)
                all_edge_attr_list.append(data_sim_edge_attr)
                cumsum_node += num_sim_nodes

                sampled_pos_node_idx = int(data.sim_node_idx[j])
                if sampled_pos_node_idx in neg_node_idx_to_sampled_idx:
                    sampled_pos_sampled_idx = neg_node_idx_to_sampled_idx[sampled_pos_node_idx]
                    batch.masked_pos_idx[j, i] = sampled_pos_sampled_idx

            for j in range(num_neg_samples):
                num_neg_nodes = data.neg_x[j].size(0)
                all_batch_list.append(torch.full((num_neg_nodes,), 1 + num_samples + j, dtype=torch.long))
                data_neg_x = data.neg_x[j]
                data_neg_edge_index = data.neg_edge_index[j]
                data_neg_edge_attr = data.neg_edge_attr[j]
                all_x_list.append(data_neg_x)
                all_edge_index_list.append(data_neg_edge_index + cumsum_node)
                all_edge_attr_list.append(data_neg_edge_attr)

                cumsum_node += num_neg_nodes

            # data.all_x = torch.cat(all_x_list, dim=0)
            # data.all_edge_index = torch.cat(all_edge_index_list, dim=1)
            # data.all_edge_attr = torch.cat(all_edge_attr_list, dim=0)
            # data.all_batch = torch.cat(all_batch_list, dim=0)
            # new_data_list.append(data)
            tmp_x_list = torch.cat(all_x_list, dim=0)
            tmp_edge_index_list = torch.cat(all_edge_index_list, dim=1)
            tmp_edge_attr_list = torch.cat(all_edge_attr_list, dim=0)
            tmp_batch_list = torch.cat(all_batch_list, dim=0)

            batch.x_list_slices.append((batch.x_list.size(0), batch.x_list.size(0) + tmp_x_list.size(0)))
            batch.edge_index_slices.append((batch.edge_index_list.size(1), batch.edge_index_list.size(1) + \
                                            tmp_edge_index_list.size(1)))
            batch.edge_attr_slices.append((batch.edge_attr_list.size(0), batch.edge_attr_list.size(0) + \
                                           tmp_edge_attr_list.size(0)))
            batch.batch_slices.append((batch.batch_list.size(0), batch.batch_list.size(0) + \
                                       tmp_batch_list.size(0)))

            batch.x_list = torch.cat([batch.x_list, tmp_x_list], dim=0)
            batch.edge_index_list = torch.cat([batch.edge_index_list, tmp_edge_index_list], dim=1)
            batch.edge_attr_list = torch.cat([batch.edge_attr_list, tmp_edge_attr_list], dim=0)
            batch.batch_list = torch.cat([batch.batch_list, tmp_batch_list], dim=0)

        batch.x_list_slices = torch.LongTensor(batch.x_list_slices)
        batch.edge_index_slices = torch.LongTensor(batch.edge_index_slices)
        batch.edge_attr_slices = torch.LongTensor(batch.edge_attr_slices)
        batch.batch_slices = torch.LongTensor(batch.batch_slices)
        # print("in batch hhh")
        # batch.data_list = new_data_list
        # print("get data list", len(batch.data_list))
        return batch
        # return batch.contiguous() # don't understand the meaning and function of batchcontiguous();...

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchContrastSimBasedExpPos(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrastSimBasedExpPos, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        with_neg = False
        if 'neg_x' in data_list[0]:
            with_neg = True

        batch = BatchContrastSimBasedExpPos()
        for key in keys:
            batch[key] = []
        batch.batch = []

        #### for positive samples
        num_samples = len(data_list[0].sim_x) # the length of the list --- number of samples
        batch.batch_sim = []
        batch.sim_x = []
        batch.sim_edge_index = []
        batch.sim_edge_attr = []
        cumsum_node_sim = 0

        # for i in range(num_samples):
        #     batch.sim_x.append([])
        #     batch.sim_edge_index.append([])
        #     batch.sim_edge_attr.append([])
        #     batch.batch_sim.append([])
        #     cumsum_node_sim.append(0)
        #
        # # num_neg_samples = len(data_list[0].neg_x)  # the length of the list --- number of samples
        # batch.batch_sim = []
        # batch.sim_x = []
        # batch.sim_edge_index = []
        # batch.sim_edge_attr = []
        # cumsum_node_sim = 0

        #### for negative samples
        if with_neg:
            num_neg_samples = len(data_list[0].neg_x)  # the length of the list --- number of samples
            batch.batch_neg = []
            batch.neg_x = []
            batch.neg_edge_index = []
            batch.neg_edge_attr = []
            cumsum_node_neg = 0
        # for i in range(num_neg_samples):
        #     batch.neg_x.append([])
        #     batch.neg_edge_index.append([])
        #     batch.neg_edge_attr.append([])
        #     batch.batch_neg.append([])
        #     cumsum_node_neg.append(0)

        cumsum_node = 0
        cumsum_edge = 0
        # cumsum_node_sim = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes

            for j in range(num_samples):
                # num_sim_nodes = data.sim_x[j].size(0)
                # batch.batch_sim[j].append(torch.full((num_sim_nodes, ), i, dtype=torch.long))
                # data_sim_x = data.sim_x[j]
                # data_sim_edge_index = data.sim_edge_index[j]
                # data_sim_edge_attr = data.sim_edge_attr[j]
                # batch.sim_x[j].append(data_sim_x)
                # batch.sim_edge_index[j].append(data_sim_edge_index + cumsum_node_sim[j])
                # batch.sim_edge_attr[j].append(data_sim_edge_attr)
                # cumsum_node_sim[j] += num_sim_nodes

                num_sim_nodes = data.sim_x[j].size(0)
                batch.batch_sim.append(torch.full((num_sim_nodes,), i * num_samples + j, dtype=torch.long))
                data_sim_x = data.sim_x[j]
                data_sim_edge_index = data.sim_edge_index[j]
                data_sim_edge_attr = data.sim_edge_attr[j]
                batch.sim_x.append(data_sim_x)
                batch.sim_edge_index.append(data_sim_edge_index + cumsum_node_sim)
                batch.sim_edge_attr.append(data_sim_edge_attr)
                cumsum_node_sim += num_sim_nodes

            if with_neg:
                for j in range(num_neg_samples):
                    num_neg_nodes = data.neg_x[j].size(0)
                    batch.batch_neg.append(torch.full((num_neg_nodes, ), i * num_neg_samples + j, dtype=torch.long))
                    data_neg_x = data.neg_x[j]
                    data_neg_edge_index = data.neg_edge_index[j]
                    data_neg_edge_attr = data.neg_edge_attr[j]
                    batch.neg_x.append(data_neg_x)
                    batch.neg_edge_index.append(data_neg_edge_index + cumsum_node_neg)
                    batch.neg_edge_attr.append(data_neg_edge_attr)

                    cumsum_node_neg += num_neg_nodes

            # num_sim_nodes = data.sim_x.size(0)
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            # batch.batch_sim.append(torch.full((num_sim_nodes, ), i, dtype=torch.long))

            # batch.batch_node_assign.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                if key in ["sim_x", "sim_edge_index", "sim_edge_attr", "neg_x", "neg_edge_index", "neg_edge_attr"]:
                    continue
                item = data[key]
                if key in ['edge_index']:
                    item = item + cumsum_node
                elif key in ['sim_edge_index']:
                    item = item + cumsum_node_sim
                batch[key].append(item)

            cumsum_node += num_nodes
            # cumsum_node_sim += num_sim_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            if key in ["sim_x", "sim_edge_index", "sim_edge_attr", "neg_x", "neg_edge_index", "neg_edge_attr"]:
                continue
            try:
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
            except:
                try:
                    batch[key] = torch.cat(batch[key], dim=1)
                except:
                    for j in batch[key]:
                        print(j.size())
                    raise RuntimeError("{}".format(str(key)))

        # for j in range(num_samples):
        #     batch.sim_x[j] = torch.cat(batch.sim_x[j], dim=0)
        #     batch.sim_edge_index[j] = torch.cat(batch.sim_edge_index[j], dim=1)
        #     batch.sim_edge_attr[j] = torch.cat(batch.sim_edge_attr[j], dim=0)
        #     batch.batch_sim[j] = torch.cat(batch.batch_sim[j], dim=-1)

        batch.sim_x = torch.cat(batch.sim_x, dim=0)
        batch.sim_edge_index = torch.cat(batch.sim_edge_index, dim=1)
        batch.sim_edge_attr = torch.cat(batch.sim_edge_attr, dim=0)
        batch.batch_sim = torch.cat(batch.batch_sim, dim=-1)

        if with_neg:
            batch.neg_x = torch.cat(batch.neg_x, dim=0)
            batch.neg_edge_index = torch.cat(batch.neg_edge_index, dim=1)
            batch.neg_edge_attr = torch.cat(batch.neg_edge_attr, dim=0)
            batch.batch_neg = torch.cat(batch.batch_neg, dim=-1)
        batch.batch = torch.cat(batch.batch, dim=-1)
        # batch.batch_sim = torch.cat(batch.batch_sim, dim=-1)

        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchFlowContrastData(Data):
    def __init__(self, batch=None):
        super(BatchFlowContrastData, self).__init__()
        self.batch = batch
    @staticmethod
    def from_data_list(data_list):

        batch = BatchMaskedData()
        batch.batch = list()
        batch.dist_batch = list()
        # batch.batch_subgraph_mapping = list()

        cumsum_node = 0
        cumsum_dist_node = 0

        keys = ["x", "dist_x", "edge_index", "dist_edge_index", "edge_attr", "dist_edge_attr"]

        for key in keys:
            batch[key] = list()

        for i, data in enumerate(data_list):
            num_atom = data.x.size(0)
            dist_num_atom = data.dist_x.size(0)
            # num_edge = data.edge_attr.size(0)
            num_split_atom = data.split_x.size(0)
            num_edge = data.pred_edge_st_ed_idx.size(1)

            batch.batch.append(torch.full((num_atom,), i, dtype=torch.long))
            batch.dist_batch.append(torch.full((dist_num_atom, ), i, dtype=torch.long))

            for key in keys:
                item = data[key]
                if key == "edge_index":
                    item = item + cumsum_node
                if key == "dist_edge_index":
                    item = item + cumsum_dist_node
                # if key in ["edge_index", "dist_edge_index"]:
                #     item = item + cumsum_node
                batch[key].append(item)
            cumsum_node += num_atom
            cumsum_dist_node += dist_num_atom

        for key in keys:
            try:
                if key in ["dist_edge_index", "edge_index"]:
                    batch[key] = torch.cat(batch[key], dim=1)
                else:
                    batch[key] = torch.cat(
                        batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
            except:
                print(key, data_list[0][key].size())
                raise RuntimeError("aa")
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.dist_batch = torch.cat(batch.dist_batch, dim=-1)
        return batch.contiguous()


class BatchFlowContrastDataVTwo(Data):
    def __init__(self, batch=None):
        super(BatchFlowContrastDataVTwo, self).__init__()
        self.batch = batch
    @staticmethod
    def from_data_list(data_list):

        batch = BatchMaskedData()
        batch.batch = list()
        batch.sim_batch = list()
        # batch.batch_subgraph_mapping = list()

        cumsum_node = 0
        cumsum_dist_node = 0
        has_local = False

        keys = ["x", "sim_x", "edge_index", "sim_edge_index", "edge_attr", "sim_edge_attr"]

        if "x_substruct" in data_list[0].keys:
            keys += ["x_substruct", "edge_index_substruct", "edge_attr_substruct", "first_approx_node_idxes", "node_type"]
            cumsum_substruct = 0
            # cumsum_first_nodes = 0
            has_local = True
            batch.first_level_node_batch = []

        for key in keys:
            batch[key] = list()

        for i, data in enumerate(data_list):
            num_atom = data.x.size(0)
            sim_num_atom = data.sim_x.size(0)

            # num_edge = data.edge_attr.size(0)
            # num_split_atom = data.split_x.size(0)
            # num_edge = data.pred_edge_st_ed_idx.size(1)

            batch.batch.append(torch.full((num_atom,), i, dtype=torch.long))
            batch.sim_batch.append(torch.full((sim_num_atom, ), i, dtype=torch.long))
            if has_local:
                num_atom_substruct = data.x_substruct.size(0)
                num_atom_first_level = data.first_approx_node_idxes.size(0)
                batch.first_level_node_batch.append(torch.full((num_atom_first_level, ), i, dtype=torch.long))

            for key in keys:
                item = data[key]
                if key == "edge_index":
                    item = item + cumsum_node
                if key == "dist_edge_index":
                    item = item + cumsum_dist_node
                # if key in ["edge_index", "dist_edge_index"]:
                #     item = item + cumsum_node
                if key in ["first_approx_node_idxes", "edge_index_substruct"]:
                    item = item + cumsum_substruct

                batch[key].append(item)
            cumsum_node += num_atom
            cumsum_dist_node += sim_num_atom
            if has_local:
                cumsum_substruct += num_atom_substruct

        for key in keys:
            try:
                if key in ["sim_edge_index", "edge_index", "edge_index_substruct"]:
                    batch[key] = torch.cat(batch[key], dim=1)
                else:
                    batch[key] = torch.cat(
                        batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
            except:
                print(key, data_list[0][key].size())
                raise RuntimeError("aa")
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.sim_batch = torch.cat(batch.sim_batch, dim=-1)
        if has_local:
            batch.first_level_node_batch = torch.cat(batch.first_level_node_batch, dim=-1)
        return batch.contiguous()


class BatchMaskedData(Data):

    def __init__(self, batch=None):
        super(BatchMaskedData, self).__init__()
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = BatchMaskedData()
        batch.batch_node = list()
        batch.batch_edge = list()
        # batch.batch_subgraph_mapping = list()

        cumsum_node = 0
        cumsum_split_node = 0
        cumsum_num_pred_edge = 0

        for key in keys:
            batch[key] = list()

        for i, data in enumerate(data_list):
            num_atom = data.x.size(0)
            # num_edge = data.edge_attr.size(0)
            num_split_atom = data.split_x.size(0)
            num_edge = data.pred_edge_st_ed_idx.size(1)

            batch.batch_node.append(torch.full((num_atom, ), i, dtype=torch.long))
            batch.batch_edge.append(torch.full((num_edge, ), i, dtype=torch.long))

            for key in data.keys:
                item = data[key]
                if key in ["edge_index", "split_node_map", "node_to_subgraph", "to_pred_edge_subgraph_idx",
                           "pred_nodes_to_node_idx", "pred_edge_node_idx"]:
                    item = item + cumsum_node
                elif key in ["split_edge_index", "to_pred_edge_st_ed_idx", "pred_nodes_node_idx", "pred_edge_st_ed_idx"]:
                    item = item + cumsum_split_node
                elif key in ["pred_edge_nodes_to_edge_idx"]:
                    item = item + cumsum_num_pred_edge
                batch[key].append(item)
            cumsum_node += num_atom
            cumsum_split_node += num_split_atom
            cumsum_num_pred_edge += num_edge

        for key in keys:
            try:
                if key in ["to_pred_edge_st_ed_idx", "split_edge_index", "pred_edge_st_ed_idx"]:
                    batch[key] = torch.cat(batch[key], dim=1)
                    # print(key, batch[key].size())
                elif key in ["y"]:
                    batch[key] = torch.cat(batch[key], dim=0)
                else:
                    batch[key] = torch.cat(
                        batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
            except:
                # print(key, data_list[0][key].size())
                raise RuntimeError("aa")
        batch.batch_node = torch.cat(batch.batch_node, dim=-1)
        batch.batch_edge = torch.cat(batch.batch_edge, dim=-1)
        return batch.contiguous()


class BatchSplitLabel(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSplitLabel, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]

        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchSplitLabel()

        for key in keys:
            batch[key] = []
        batch.batch = []

        batch.batch = []
        batch.batch_pos = []
        batch.batch_neg = []

        batch.edge_index = []
        batch.edge_index_pos = []
        batch.edge_index_neg = []

        batch.edge_attr = []
        batch.edge_attr_pos = []
        batch.edge_attr_neg = []

        batch.x = []
        batch.x_pos = []
        batch.x_neg = []

        keys = keys + ["edge_index_pos", "edge_index_neg", "edge_attr_pos", "edge_attr_neg", "x_pos", "x_neg"]

        cumsum_node = 0
        cumsum_edge = 0

        cumsum_node_pos = 0
        cumsum_node_neg = 0
        cumsum_edge_pos = 0
        cumsum_edge_neg = 0

        MOD = 3

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            if i % MOD == 0:
                batch.batch.append(torch.full((num_nodes, ), i // MOD, dtype=torch.long))
            elif i % MOD == 1:
                batch.batch_pos.append(torch.full((num_nodes, ), i // MOD, dtype=torch.long))
            else:
                batch.batch_neg.append(torch.full((num_nodes, ), i // MOD, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key == "edge_index":
                    if i % MOD == 0:
                        item = item + cumsum_node
                        batch.edge_index.append(item)
                        cumsum_node += num_nodes
                    elif i % MOD == 1:
                        item = item + cumsum_node_pos
                        batch.edge_index_pos.append(item)
                        cumsum_node_pos += num_nodes
                    else:
                        item = item + cumsum_node_neg
                        batch.edge_index_neg.append(item)
                        cumsum_node_neg += num_nodes
                elif key == "edge_attr":
                    if i % MOD == 0:
                        batch.edge_attr.append(item)
                    elif i % MOD == 1:
                        batch.edge_attr_pos.append(item)
                    else:
                        batch.edge_attr_neg.append(item)
                elif key == "x":
                    if i % MOD == 0:
                        batch.x.append(item)
                    elif i % MOD == 1:
                        batch.x_pos.append(item)
                    else:
                        batch.x_neg.append(item)
                elif key == "connected_edge_indices":
                    if i % MOD == 0:
                        item = item + cumsum_edge
                        cumsum_edge += data.edge_index.shape[1]
                    elif i % MOD == 1:
                        item = item + cumsum_edge_pos
                        cumsum_edge_pos += data.edge_index.shape[1]
                    else:
                        item = item + cumsum_edge_neg
                        cumsum_edge_neg += data.edge_index.shape[1]
                    batch[key].append(item)
                else:
                    batch[key].append(item)

        for key in keys:
            if key == "wave_emb":
                batch[key] = torch.cat(batch[key], dim=0)
            else:
                try:
                    batch[key] = torch.cat(
                        batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
                except:
                    try:
                        batch[key] = torch.cat(batch[key], dim=1)
                    except:
                        for j in batch[key]:
                            print(j.size())
                        raise RuntimeError("{}".format(str(key)))
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_pos = torch.cat(batch.batch_pos, dim=-1)
        batch.batch_neg = torch.cat(batch.batch_neg, dim=-1)

        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchContrast(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrast, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        batch_num_nodes = sum([data.num_nodes for data in data_list])
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()
        # batch.batch_node_assign = []
        # batch_keys = ["batch_node_assign"]
        for key in keys:
            batch[key] = []
        batch.batch = []
        batch.batch_a = []
        batch.batch_b = []

        cumsum_node = 0
        cumsum_edge = 0
        cumsum_node_a = 0
        cumsum_node_b = 0
        batch.num_nodes_a = []
        batch.num_nodes_b = []
        batch.num_edges_a, batch.num_edges_b = [], []
        # batch.mask = []
        batch.num_moleculor = len(data_list)
        # alli = {}
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            n_a = data.x_a.size(0)
            n_b = data.x_b.size(0)
            batch.batch_a.append(torch.full((n_a, ), i, dtype=torch.long))
            batch.batch_b.append(torch.full((n_b, ), i, dtype=torch.long))
            batch.num_nodes_a.append(n_a)
            batch.num_nodes_b.append(n_b)

            # batch.batch_node_assign.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices', "masked_atom_indices_a", "masked_atom_indices_b", 'edge_index_a', 'edge_index_b']:

                    if key == "edge_index_a":
                        item = item + cumsum_node_a
                    elif key == "edge_index_b":
                        item = item + cumsum_node_b
                    else:
                        item = item + cumsum_node
                elif key == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            # data.mask = torch.zeros((num_nodes, batch_num_nodes))
            # node_tensor = torch.arange(num_nodes) + cumsum_node
            # # print(i, num_nodes, cumsum_node)
            # # print("mask_shape1 ", data.mask.size())
            # data.mask[:, node_tensor] = 1.
            # # print("mask_shape2 ", data.mask.size())
            # batch.mask.append(data.mask)
            cumsum_node += num_nodes
            cumsum_node_a += n_a
            cumsum_node_b += n_b

            cumsum_edge += data.edge_index.shape[1]
            batch.num_edges_a.append(data.edge_index_a.shape[1])
            batch.num_edges_b.append(data.edge_index_b.shape[1])


        for key in keys:
            if key == "wave_emb":
                batch[key] = torch.cat(batch[key], dim=0)
            else:
                # try:
                if key == "filter_k_a" or key == "filter_k_b":
                    batch[key] = torch.LongTensor(batch[key])
                else:
                    batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
                # except:
                #     print(key)
                #     raise RuntimeError("invalid argument 0")
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_a = torch.cat(batch.batch_a, dim=-1)
        batch.batch_b = torch.cat(batch.batch_b, dim=-1)
        batch.num_nodes_a = torch.LongTensor(batch.num_nodes_a)
        batch.num_nodes_b = torch.LongTensor(batch.num_nodes_b)
        batch.num_edges_a = torch.LongTensor(batch.num_edges_a)
        batch.num_edges_b = torch.LongTensor(batch.num_edges_b)

        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices', 'edge_index_a', 'edge_index_b']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchContrast2(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrast2, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        batch_num_nodes = sum([data.num_nodes for data in data_list])
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()
        # batch.batch_node_assign = []
        # batch_keys = ["batch_node_assign"]
        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0
        # batch.mask = []
        batch.num_moleculor = len(data_list)
        # alli = {}
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))

            # batch.batch_node_assign.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices', "masked_atom_indices_a", "masked_atom_indices_b", "s_nodes"]:

                    item = item + cumsum_node
                elif key == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes

            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            if key == "wave_emb":
                batch[key] = torch.cat(batch[key], dim=0)
            else:
                if key == "k" :
                    continue
                batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.k = data_list[0].k
        batch.batch = torch.cat(batch.batch, dim=-1)

        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices', 'edge_index_a', 'edge_index_b']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchContrast3(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrast3, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        batch_num_nodes = sum([data.num_nodes for data in data_list])
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()
        # batch.batch_node_assign = []
        # batch_keys = ["batch_node_assign"]
        for key in keys:
            batch[key] = []
        batch.batch = []
        batch.batch_a = []
        batch.batch_b = []

        cumsum_node = 0
        cumsum_edge = 0
        cumsum_node_a = 0
        cumsum_node_b = 0
        # batch.num_nodes_a = []
        # batch.num_nodes_b = []
        # batch.num_edges_a, batch.num_edges_b = [], []
        # batch.mask = []
        batch.num_moleculor = len(data_list)
        # alli = {}
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            n_a = data.x_a.size(0)
            n_b = data.x_b.size(0)
            batch.batch_a.append(torch.full((n_a, ), i, dtype=torch.long))
            batch.batch_b.append(torch.full((n_b, ), i, dtype=torch.long))
            # batch.num_nodes_a.append(n_a)
            # batch.num_nodes_b.append(n_b)

            # batch.batch_node_assign.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices', "masked_atom_indices_a", "masked_atom_indices_b", 'edge_index_a', 'edge_index_b', "masked_atom_indices_a", "masked_atom_indices_b"]:

                    if key == "edge_index_a" or key == "masked_atom_indices_a":
                        item = item + cumsum_node_a
                    elif key == "edge_index_b" or key == "masked_atom_indices_b":
                        item = item + cumsum_node_b
                    else:
                        item = item + cumsum_node
                elif key == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            # data.mask = torch.zeros((num_nodes, batch_num_nodes))
            # node_tensor = torch.arange(num_nodes) + cumsum_node
            # # print(i, num_nodes, cumsum_node)
            # # print("mask_shape1 ", data.mask.size())
            # data.mask[:, node_tensor] = 1.
            # # print("mask_shape2 ", data.mask.size())
            # batch.mask.append(data.mask)
            cumsum_node += num_nodes
            cumsum_node_a += n_a
            cumsum_node_b += n_b

            cumsum_edge += data.edge_index.shape[1]
            # batch.num_edges_a.append(data.edge_index_a.shape[1])
            # batch.num_edges_b.append(data.edge_index_b.shape[1])


        for key in keys:
            if key == "wave_emb":
                batch[key] = torch.cat(batch[key], dim=0)
            else:
                # try:
                if key == "filter_k_a" or key == "filter_k_b":
                    batch[key] = torch.LongTensor(batch[key])
                else:
                    batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
                # except:
                #     print(key)
                #     raise RuntimeError("invalid argument 0")
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_a = torch.cat(batch.batch_a, dim=-1)
        batch.batch_b = torch.cat(batch.batch_b, dim=-1)
        # batch.num_nodes_a = torch.LongTensor(batch.num_nodes_a)
        # batch.num_nodes_b = torch.LongTensor(batch.num_nodes_b)
        # batch.num_edges_a = torch.LongTensor(batch.num_edges_a)
        # batch.num_edges_b = torch.LongTensor(batch.num_edges_b)

        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices', 'edge_index_a', 'edge_index_b']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchContrast4(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrast4, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        batch_num_nodes = sum([data.num_nodes for data in data_list])
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()
        # batch.batch_node_assign = []
        # batch_keys = ["batch_node_assign"]
        for key in keys:
            batch[key] = []
        batch.batch = []
        batch.batch_a = []
        batch.batch_b = []

        cumsum_node = 0
        cumsum_edge = 0
        batch.num_moleculor = len(data_list)
        # alli = {}
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            n_a = data.a_nodes.size(0)
            n_b = data.b_nodes.size(0)
            batch.batch_a.append(torch.full((n_a, ), i, dtype=torch.long))
            batch.batch_b.append(torch.full((n_b, ), i, dtype=torch.long))
            # batch.num_nodes_a.append(n_a)
            # batch.num_nodes_b.append(n_b)

            # batch.batch_node_assign.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices', "a_nodes", "b_nodes"]:

                    item = item + cumsum_node
                elif key == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes

            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            if key == "wave_emb":
                batch[key] = torch.cat(batch[key], dim=0)
            else:
                # try:
                if key == "filter_k_a" or key == "filter_k_b":
                    batch[key] = torch.LongTensor(batch[key])
                else:
                    batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
                # except:
                #     print(key)
                #     raise RuntimeError("invalid argument 0")
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_a = torch.cat(batch.batch_a, dim=-1)
        batch.batch_b = torch.cat(batch.batch_b, dim=-1)

        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices', 'edge_index_a', 'edge_index_b']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchContrast5(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrast5, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        batch_num_nodes = sum([data.num_nodes for data in data_list])
        keys = list(set.union(*keys))
        assert 'batch' not in keys


        batch = BatchMasking()
        # batch.batch_node_assign = []
        # batch_keys = ["batch_node_assign"]
        for key in keys:
            batch[key] = []
        batch.batch = []
        # batch.batch_a = []
        # batch.batch_b = []
        batch.context_nodes = []

        cumsum_subgraph = 0

        cumsum_node = 0
        cumsum_edge = 0
        batch.num_moleculor = len(data_list)
        # alli = {}
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))

            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices', "center_nodes", "context_nodes"]:
                    item = item + cumsum_node
                elif key == "context_nodes_to_graph_idx":
                    item = item + cumsum_subgraph

                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_subgraph += data.k
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            if key == "k":
                batch[key] = batch[key][0]
            else:
                batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.context_nodes_to_graph_idx = batch.context_nodes_to_graph_idx.long()

        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices', 'edge_index_a', 'edge_index_b']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchContrast6(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrast6, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        batch_num_nodes = sum([data.num_nodes for data in data_list])
        keys = list(set.union(*keys))
        assert 'batch' not in keys


        batch = BatchMasking()
        # batch.batch_node_assign = []
        # batch_keys = ["batch_node_assign"]
        for key in keys:
            if key == "dis_to_nodes":
                continue
            batch[key] = []
        batch.batch = []

        batch.context_nodes = []

        batch.k = data_list[0].k
        batch.dis_to_nodes_fr = {i: [] for i in range(batch.k)}
        batch.dis_to_nodes_to = {i: [] for i in range(batch.k)}
        batch.cumsum_k = {i: 0 for i in range(batch.k)}
        batch.nodes_fr_to_subgraph = {i: [] for i in range(batch.k)}
        batch.nodes_to_to_subgraph = {i: [] for i in range(batch.k)}

        cumsum_subgraph = 0

        cumsum_node = 0
        cumsum_edge = 0
        batch.num_moleculor = len(data_list)
        # alli = {}
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))

            for key in data.keys:
                if key == "k":
                    continue
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices', "center_nodes", "context_nodes"]:
                    item = item + cumsum_node
                elif key == "context_nodes_to_graph_idx":
                    item = item + cumsum_subgraph
                elif key == "dis_to_nodes":
                    for j in range(len(data.dis_to_nodes) - 1):
                        batch.dis_to_nodes_fr[j].append(data.dis_to_nodes[j] + cumsum_node)
                        batch.dis_to_nodes_to[j].append(data.dis_to_nodes[j + 1] + cumsum_node)
                        batch.nodes_fr_to_subgraph[j].append(torch.full((len(data.dis_to_nodes[j]),), batch.cumsum_k[j], dtype=torch.long))
                        batch.nodes_to_to_subgraph[j].append(torch.full((len(data.dis_to_nodes[j + 1]),), batch.cumsum_k[j], dtype=torch.long))
                        batch.cumsum_k[j] += 1
                if key != "dis_to_nodes":
                    batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_subgraph += data.k
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            if key == "k":
                continue
            elif key != "dis_to_nodes":
                batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch = torch.cat(batch.batch, dim=-1)
        # batch.context_nodes_to_graph_idx = batch.context_nodes_to_graph_idx.long()
        for j in range(batch.k):
            batch.dis_to_nodes_fr[j] = torch.cat(batch.dis_to_nodes_fr[j], dim=0)
            batch.dis_to_nodes_to[j] = torch.cat(batch.dis_to_nodes_to[j], dim=0)
            batch.nodes_fr_to_subgraph[j] = torch.cat(batch.nodes_fr_to_subgraph[j], dim=0)
            batch.nodes_to_to_subgraph[j] = torch.cat(batch.nodes_to_to_subgraph[j], dim=0)

            # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices', 'edge_index_a', 'edge_index_b']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchContrast7(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchContrast7, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        batch_num_nodes = sum([data.num_nodes for data in data_list])
        keys = list(set.union(*keys))
        assert 'batch' not in keys


        batch = BatchMasking()
        # batch.batch_node_assign = []
        # batch_keys = ["batch_node_assign"]
        for key in keys:
            if key == "dis_to_nodes":
                continue
            batch[key] = []
        batch.batch = []

        batch.context_nodes = []

        cumsum_subgraph = 0

        cumsum_node = 0
        cumsum_edge = 0
        batch.num_moleculor = len(data_list)
        cumsum_subgraphs = 0
        # alli = {}
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))

            for key in data.keys:
                if key == "k":
                    continue
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices', "center_nodes", "context_nodes"]:
                    item = item + cumsum_node
                elif key == "context_nodes_to_graph_idx":
                    item = item + cumsum_subgraph
                elif key == "context_nodes_to_idx":
                    item = item + cumsum_subgraphs

                if key != "dis_to_nodes":
                    batch[key].append(item)

            cumsum_node += num_nodes
            # cumsum_subgraph += data.k
            cumsum_subgraphs += data.masked_atom_num
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            if key == "k" or key == "masked_atom_num":
                continue
            elif key != "dis_to_nodes":
                batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch = torch.cat(batch.batch, dim=-1)
        # batch.context_nodes_to_graph_idx = batch.context_nodes_to_graph_idx.long()

            # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices', 'edge_index_a', 'edge_index_b']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchMasSubstruct(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasSubstruct, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        batch_num_nodes = sum([data.num_nodes for data in data_list])
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()
        # batch.batch_node_assign = []
        # batch_keys = ["batch_node_assign"]
        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0
        # batch.mask = []
        batch.num_moleculor = len(data_list)
        # alli = {}
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))

            # batch.batch_node_assign.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'edge_indices_masked', "cut_order_indices", "mask_node_indices"]:
                    item = item + cumsum_node
                elif key  == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            # data.mask = torch.zeros((num_nodes, batch_num_nodes))
            # node_tensor = torch.arange(num_nodes) + cumsum_node
            # # print(i, num_nodes, cumsum_node)
            # # print("mask_shape1 ", data.mask.size())
            # data.mask[:, node_tensor] = 1.
            # # print("mask_shape2 ", data.mask.size())
            # batch.mask.append(data.mask)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            try:
                if key == "edge_indices_masked":
                    batch[key] = torch.cat(batch[key], dim=1)
                else:
                    batch[key] = torch.cat(
                        batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
            except:
                print(key)
                raise RuntimeError("aaa")
        batch.batch = torch.cat(batch.batch, dim=-1)
        # batch.mask = torch.cat(batch.mask, dim=0)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchAE(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchAE, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'negative_edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0


class BatchSubstructContext(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        #keys = [set(data.keys) for data in data_list]
        #keys = list(set.union(*keys))
        #assert 'batch' not in keys

        cona = False

        batch = BatchSubstructContext()
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct",
                "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context", "first_approx_node_idxes", "node_type"]

        one_data_keys = list(data_list[0].keys)

        if "x_masked" in one_data_keys:
            keys += ["x_masked", "edge_attr_masked", "edge_index", "x", "edge_attr"]
            cona = True

        for key in keys:
            #print(key)
            batch[key] = []

        #batch.batch = []
        #used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        # add for first level node type prediction
        batch.first_level_node_batch = []

        if cona == True:
            batch.batch = list()


        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0


        i = 0
        
        for data in data_list:
            #If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                #batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                # add for first level node type prediction
                batch.first_level_node_batch.append(torch.full((len(data.first_approx_node_idxes), ), i, dtype=torch.long))

                if cona == True:
                    batch.batch.append(torch.full((data.num_nodes, ), i, dtype=torch.long))

                ###batching for the main graph
                #for key in data.keys:
                #    if not "context" in key and not "substruct" in key:
                #        item = data[key]
                #        item = item + cumsum_main if batch.cumsum(key, item) else item
                #        batch[key].append(item)
                
                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)

                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                for key in ["node_type", "first_approx_node_idxes"]:
                    item = data[key]
                    if key == "first_approx_node_idxes":
                        item = item + cumsum_substruct
                    batch[key].append(item)

                if cona == True:
                    for key in ["edge_index"]:
                        item = data[key]
                        item = item + cumsum_main
                        batch[key].append(item)
                    for key in ["x", "x_masked", "edge_attr", "edge_attr_masked"]:
                        item = data[key]
                        batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct   
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            try:
                if key in ["x", "x_masked", "edge_attr", "edge_attr_masked"]:
                    if (batch[key][0].size(1) != batch[key][1].size(1) or batch[key][0].size(1) != 2):
                        # print(key, batch[key][0].size(), batch[key][1].size())
                        batch[key] = torch.cat(batch[key], dim=1)
                        batch[key] = torch.transpose(batch[key], 0, 1)
                    else:
                        # print(key, batch[key][0].size(), batch[key][1].size())
                        batch[key] = torch.cat(batch[key], dim=0)
                elif key in ["edge_index"]:
                    if (batch[key][0].size(0) != batch[key][1].size(0)) or (batch[key][0].size(0) != 2):
                        batch[key] = torch.cat(batch[key], dim=0)
                        batch[key] = torch.transpose(batch[key], 0, 1)
                    else:
                        batch[key] = torch.cat(batch[key], dim=1)
                else:
                    batch[key] = torch.cat(
                        batch[key], dim=batch.cat_dim(key))
                # print(key, batch[key][0].size())
            except:
                # if key in ["x", "x_masked", "edge_attr", "edge_attr_masked"]:
                #     batch[key] = torch.cat(batch[key], dim=1)
                #     batch[key] = torch.transpose(batch[key], 0, 1)
                # elif key in ["edge_index"]:
                #     batch[key] = torch.cat(batch[key], dim=0)
                #     batch[key] = torch.transpose(batch[key], 0, 1)
                # else:
                raise RuntimeError("Cannot concatenate the key {}".format(key))
                # print(key, batch[key][0].size(), batch[key][1].size())

        #batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)
        batch.first_level_node_batch = torch.cat(batch.first_level_node_batch,  dim=-1)
        if cona == True:
            batch.batch = torch.cat(batch.batch, dim=-1)
        # with open("log.txt", "a") as wf:
        #     for i in range(batch.first_level_node_batch.size(0)):
        #         wf.write("{}\n".format(str(batch.first_level_node_batch[i])))

        return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1



class BatchContrastSubgraph(Data):

    def __init__(self, batch=None, **kwargs):
        super(BatchContrastSubgraph, self).__init__(**kwargs)
        self.batch = batch
        # self.atom_vocab = atom_vocab

    @staticmethod
    def from_data_list(data_list):

        # coar = False
        keys = [set(data.keys) for data in data_list]
        # batch_num_nodes = sum([data.num_nodes for data in data_list])
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchContrastSubgraph()

        for key in keys:
            batch[key] = []
        batch.batch_a = []
        batch.batch_b = []


        cumsum_node_a = 0
        cumsum_node_b = 0
        for i, data in enumerate(data_list):
            # num_nodes = data.num_nodes
            na = data.x_a.size(0)
            nb = data.x_b.size(0)
            batch.batch_a.append(torch.full((na, ), i, dtype=torch.long))
            batch.batch_b.append(torch.full((nb, ), i, dtype=torch.long))

            for key in data.keys:
                item = data[key]
                if key in ['edge_index', "edge_index_a", "edge_index_b"]:
                    if key == "edge_index_a":
                        item += na
                    elif key == "edge_index_b":
                        item += nb
                batch[key].append(item)

            cumsum_node_a += na
            cumsum_node_b += nb

        for key in keys:
            if key in ['edge_index', "edge_index_a", "edge_index_b"]:
                batch[key] = torch.cat(batch[key], dim=1)
            else:
                batch[key] = torch.cat(batch[key], dim=0)
        batch.batch_a = torch.cat(batch.batch_a, dim=-1)
        batch.batch_b = torch.cat(batch.batch_b, dim=-1)

        return batch.contiguous()

    def cumsum(self, key, item):
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
