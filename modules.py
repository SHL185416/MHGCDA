import numpy as np
import torch
import torch.nn.functional as F
from utils import one_hot_encode_torch, calculate_pred_label_t, sim


def transferability_weights(num_source, pred_logit_t_list):
    r"""transferability weights based on information entropy.

    Transferability weights are calculated by the information entropy of the predicted logits.

    :arg
        num_source (int): number of source domains.
        pred_logit_t_list (list): predicted logits from target domain.
    :return
        w_k_list (list): transferability weights for each source domain.
    """
    w_k_list = [0] * num_source
    for i in range(num_source):
        entropy = (-F.softmax(pred_logit_t_list[i].detach(), dim=1) * torch.log_softmax(
            pred_logit_t_list[i].detach(),
            dim=1)).sum(1)
        w_k_list[i] = 1 / entropy.mean()
    w_k_sum = sum(w_k_list)
    for i in range(num_source):
        w_k_list[i] = w_k_list[i] / w_k_sum
    return w_k_list


def pseudo_labeling(pred_logit_t_list, tau_p, tau_n, w_k_list=None):
    r"""positive and negative pseudo-labeling.

    Using the logit of the target node for positive and negative pseudo label learning。

    :arg
        pred_logit_t_list (list): predicted logits from target domain.
        tau_p (float): threshold for positive pseudo-labeling.
        tau_n (float): threshold for negative pseudo-labeling.
        w_k_list (list): transferability weights for each source domain.
    :return
        loss: pseudo labeling loss
    """
    num_source = len(w_k_list)
    indices_list = [0] * num_source
    pred_label_clf_list = [0] * num_source
    # 生成分类器clf_0/1和pred_Y_t_kmean一致的伪标签
    for j in range(len(pred_logit_t_list)):
        _, indices_list[j] = torch.max(pred_logit_t_list[j], dim=1)
        pred_label_clf_list[j] = one_hot_encode_torch(indices_list[j], pred_logit_t_list[j].shape[1])

    pred_label_pl = calculate_pred_label_t(pred_label_clf_list).float()
    pred_label_pl = pred_label_pl.to(indices_list[0].device)
    pred_label_nl = 0
    for i in range(len(pred_label_clf_list)):
        pred_label_nl += pred_label_clf_list[i]
    pred_label_nl = pred_label_nl.to(indices_list[0].device)

    loss_ce = 0
    loss_nl = 0

    # 使用激活函数
    pred_logit_node_t_softmax_list = [0] * num_source
    for j in range(num_source):
        pred_logit_node_t_softmax_list[j] = F.softmax(pred_logit_t_list[j], dim=1)
        pred_logit_node_t_softmax_list[j] = pred_logit_node_t_softmax_list[j].to(indices_list[0].device)

    # positive learning
    for i in range(0, pred_logit_t_list[0].shape[1]):
        # 找出置信度高于阈值的节点
        positive_idx_list = [0] * num_source
        for j in range(num_source):
            positive_idx_list[j] = np.array(
                torch.where((pred_logit_node_t_softmax_list[j][:, i] >= tau_p) * (pred_label_pl[:, i] > 0.0))[0].cpu())
        # 找出两个数组的交集
        positive_idx = positive_idx_list[0]
        for k in range(num_source):
            positive_idx = np.intersect1d(positive_idx, positive_idx_list[k])
        # print("正伪标签的个数",positive_idx.shape)
        for j in range(num_source):
            if positive_idx.size > 0:
                loss_ce += w_k_list[j] * F.cross_entropy(pred_logit_t_list[j][positive_idx],
                                                         indices_list[j][positive_idx], reduction='mean')

    # negative learning
    for i in range(0, pred_logit_t_list[0].shape[1]):
        # 找出置信度低于阈值的节点
        nl_idx_list = [0] * num_source
        for j in range(num_source):
            nl_idx_list[j] = np.array(
                torch.where((pred_logit_node_t_softmax_list[j][:, i] <= tau_n) * (pred_label_nl[:, i] == 0.0))[0].cpu())

        # 找出两个数组的交集
        nl_idx = nl_idx_list[0]
        for k in range(num_source):
            nl_idx = np.intersect1d(nl_idx, nl_idx_list[k])
        # print("负伪标签的个数",nl_idx.shape)
        for j in range(num_source):
            if nl_idx.size > 0:
                nl_logits = pred_logit_t_list[j][nl_idx]
                # 得到\hat{y}
                pred_nl = F.softmax(nl_logits, dim=1)
                # 得到1-\hat{y}
                pred_nl = 1 - pred_nl
                pred_nl = torch.clamp(pred_nl, 1e-7, 1.0)
                loss_nl += w_k_list[j] * (-torch.sum(torch.log(pred_nl[:, i]), dim=-1) / (nl_idx.size + 1e-7))

    return loss_nl + loss_ce


def node_level_graph_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, tau, inter_adj, pred_label_t,
                                      is_source_anchor, hidden_norm: bool = True):
    r"""node-level graph contrastive loss.

    Calculate node-level graph contrastive loss based on node embeddings in the source and target networks.

    :arg
        z1,z2: node embeddings in the source network Or target network, shape: (batch_size, embed_dim)
        tau: temperature parameter
        inter_adj: adjacency matrix of the graph, shape: (batch_size, batch_size)
        pred_label_t: prediction labels in the target network, shape: (batch_size, num_classes)
        is_source_anchor: boolean tensor, shape: (1)
    :return
        loss: node-level graph contrastive loss
    """
    inter_adj[inter_adj > 0] = 1
    target_label_indicator = pred_label_t.sum(1)
    target_label_indicator[target_label_indicator > 0] = 1
    f = lambda x: torch.exp(x / tau)
    between_sim = f(sim(z1, z2, hidden_norm))
    molecule = (between_sim.mul(inter_adj)).sum(1)
    nei_count = torch.sum(inter_adj, 1)
    nei_count = torch.squeeze(nei_count.clone().detach())
    if is_source_anchor:
        denominator = (between_sim.mul(target_label_indicator)).sum(1)
    else:
        denominator = between_sim.sum(1)
    denominator = denominator[molecule != 0]
    nei_count = nei_count[molecule != 0]
    molecule = molecule[molecule != 0]
    if len(molecule) != 0:
        loss = molecule / denominator
        loss = loss / nei_count
        return -torch.log(loss)
    return torch.Tensor([0])


def prototype_level_graph_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, tau, O_inter, O_intra,
                                           hidden_norm: bool = True):
    r"""prototype-level graph contrastive loss.

    Calculate prototype-level graph contrastive loss based on node prototypes in the source and target networks.

    :arg
        z1,z2: source node prototypes Or target node prototypes.
        tau: temperature parameter.
        O_inter: inter-class similarity matrix.
        O_intra: intra-class similarity matrix.
        hidden_norm: whether to normalize the hidden layer.
    :return
        loss: prototype-level graph contrastive loss.
    """
    O_intra = O_intra - (O_intra.diag().diag())
    f = lambda x: torch.exp(x / tau)
    inter_sim_st = f(sim(z1, z2, hidden_norm))
    intra_sim_ss = f(sim(z1, z1, hidden_norm))
    molecule = inter_sim_st.diag() * O_inter.diag()
    denominator = (inter_sim_st * O_inter).sum(1) + (intra_sim_ss * O_intra).sum(1)
    denominator = denominator[molecule > 0]
    molecule = molecule[molecule > 0]
    if len(molecule) != 0:
        loss = molecule / denominator
        return -torch.log(loss)
    return torch.Tensor([0])
