import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab


# =========================
# InsDis and MoCo
# =========================

class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryInsDis, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, x.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        return out


class MemoryMolecular(nn.Module):

    def __init__(self, feature_size, K):
        super(MemoryMolecular, self).__init__()
        # buffer will not be updated....
        self.queue_size = K
        self.index = 0
        self.feature_size = feature_size
        stdv = 1. / math.sqrt(feature_size / 3)
        self.register_buffer("feature_queue", torch.rand(K, feature_size).mul_(2 * stdv).add(-stdv) )

        self.register_buffer("rep_queue", torch.rand(K, feature_size).mul_(2 * stdv).add(-stdv) )

    def get_similar_unsimilar_data_list(self, x):
        # x.size() = bs x feature_size
        with torch.no_grad():
            logits = torch.matmul(x, self.feature_queue)
            pos_idx = torch.argmax(logits, dim=-1, keepdim=False)
            neg_idx = torch.argmax(-logits, dim=-1, keepdim=False)
            pos_rep = self.rep_queue(torch.arange(0, x.size(0)).to(x.device), pos_idx)
            neg_rep = self.rep_queue(torch.arange(0, x.size(0)).to(x.device), neg_idx)
            return pos_rep, neg_rep

    def put_in_queue(self, x):
        # should make sure that x.size(0) < self.queue_size
        with torch.no_grad():
            out_ids = torch.arange(x.size(0)).to(x.device)
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queue_size)
            out_ids = out_ids.long()
            self.feature_queue.index_copy_(0, out_ids, x)
            self.index = (self.index + x.size(0)) % self.queue_size


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out

class MemoryMolMasks(nn.Module):
    """Fixed-size queue with momentum encoder""" # for the profit: profit + ratio * sqrt(ln(totVisNum) / thisVisNum)
    def __init__(self, totBatchNum:int, queueSize:int=50, maxBatchNodes:int=10000, ratio=0.95):
        super(MemoryMolMasks, self).__init__()

        self.totBatchNum = totBatchNum
        self.queueSize = queueSize
        self.maxBatchNodes = maxBatchNodes
        self.ratio = ratio

        #
        self.register_buffer("masked_nodes_idx", torch.zeros((self.totBatchNum, self.queueSize, self.maxBatchNodes),
                                                             dtype=torch.long))
        # every time we pus things into the queue, we need to push them starting from those indexes.
        self.register_buffer("queue_st_idx", torch.zeros((self.totBatchNum, ), dtype=torch.long))
        # current queue size for the buffered queue; they are useful when we calculate which one should be adopted as the masked version.
        self.register_buffer("queue_size", torch.zeros((self.totBatchNum, ), dtype=torch.long))
        self.register_buffer("num_masked_nodes", torch.zeros((self.totBatchNum, self.queueSize), dtype=torch.long))

        self.register_buffer("mocked_times", torch.ones((self.totBatchNum, self.queueSize), dtype=torch.float64))
        self.register_buffer("mocked_profits", torch.zeros((self.totBatchNum, self.queueSize), dtype=torch.float64))


    def push_into_queue(self, batch_idx, masked_nodes_idx, num_masked_nodes):
        # masked_nodes_idx: to_put_in_que_size x num_masked_node
        # device = masked_nodes_idx.device

        num_masked_version = masked_nodes_idx.size(0)
        max_num_masked_nodes_this_batch = masked_nodes_idx.size(1)
        in_queue_idx = torch.arange(num_masked_version)
        in_queue_idx += self.queue_st_idx[batch_idx]
        in_queue_idx = torch.fmod(in_queue_idx, self.queueSize)

        self.masked_nodes_idx[batch_idx, in_queue_idx, :max_num_masked_nodes_this_batch] = masked_nodes_idx.clone()
        self.num_masked_nodes[batch_idx, in_queue_idx] = num_masked_nodes
        # reset mocked-times and mocked-profits
        self.mocked_times[batch_idx, in_queue_idx] = 1.0
        self.mocked_profits[batch_idx, in_queue_idx] = 0.0
        # update queue start index
        self.queue_st_idx[batch_idx] = (self.queue_st_idx[batch_idx] + num_masked_version) % self.queueSize
        # update queue size
        self.queue_size[batch_idx] += num_masked_version
        self.queue_size = torch.where(self.queue_size < self.queueSize, self.queue_size,
                                      torch.full((self.totBatchNum, ), self.queueSize - 1, dtype=torch.long))

    def get_best_masked_node_idx(self, batch_idx, epoch):
        queue_size = self.queue_size[batch_idx]
        mocked_times = self.mocked_times[batch_idx, :queue_size + 1]
        mocked_profits = self.mocked_profits[batch_idx, :queue_size + 1]
        overall_profits = mocked_profits + self.ratio * torch.sqrt(torch.log(epoch) / mocked_times)
        target_version_idx = overall_profits.argmax(dim=0, keepdim=False)
        # print(target_version_idx.size())
        masked_nodes_idx = self.masked_nodes_idx[batch_idx, target_version_idx, :self.num_masked_nodes[batch_idx, target_version_idx]]
        return target_version_idx, masked_nodes_idx

    def renew_mocked_info(self, batch_idx, masked_version_idx, mocked_profits):
        self.mocked_times[batch_idx, masked_version_idx] += 1.0
        self.mocked_profits[batch_idx, masked_version_idx] += mocked_profits

