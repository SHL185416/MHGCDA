from torch import nn
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    r""" GNN encoder

    Description
    -----------
    GAT model from `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__.
    GNN encoder f_h to generate node embeddings for each network.

    Parameters
    ----------
    num_layers : int
        Number of GAT layers.
    in_dim : int
        Input feature dimension.
    num_hidden : int
        Number of hidden units.
    heads : int
        Number of attention heads.
    activation : callable activation function/layer or None
        If not None, apply an activation function to the updated node features.
    feat_drop : float, optional
        Dropout rate on the input feature of each GAT layer.
    attn_drop : float, optional
        Dropout rate on the attention score of each GAT layer.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    """

    def __init__(self, num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))

    def forward(self, inputs, g):
        h = inputs
        for layer in range(self.num_layers):
            temp = h.flatten(1)
            h = self.gat_layers[layer](g, temp, get_attention=True)[0]
        return h


class NodeClassifier(nn.Module):
    r""" Node classification layer.

    Description
    -----------
    multiple unshared-weight network-specific node classifiers {f_y^((k) ) }_(k=1)^K are added to adapt to various data
    distributions of different source networks.

    Parameters
    ----------
    num_hidden : int
        The number of hidden units in each layer.
    heads : list of int
        The number of heads in each layer.
    feat_drop : float
        Dropout rate on the input feature.
    attn_drop : float
        Dropout rate on the attention weight.
    negative_slope : float
        The negative slope used in LeakyReLU.
    residual : bool
        If True, use residual connection.
    num_classes : int
        The number of classes for prediction.
    num_source : int
        The number of source networks.
    """

    def __init__(self, num_hidden, heads, feat_drop, attn_drop, negative_slope, residual, num_classes, num_source):
        super(NodeClassifier, self).__init__()
        self.num_source = num_source
        self.node_classifier_list = nn.ModuleList()

        for i in range(num_source):
            self.node_classifier_list.append(GATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, h_list, g_list, is_target=False):
        logits_list = []
        if is_target:
            for i in range(self.num_source):
                logits_list.append(self.node_classifier_list[i](g_list, h_list.flatten(1)).mean(1))
        else:
            for i in range(self.num_source):
                logits_list.append(self.node_classifier_list[i](g_list[i], h_list[i].flatten(1)).mean(1))
        return logits_list


class MHGCDA(nn.Module):
    r""" Hierarchical Graph Contrastive Domain Adaptation for Multi-source Cross-network Node Classification

    Description:
    ----------
    The model architecture of MHGCDA, which contains a shared-weight GNN encoder along
    with multiple unshared-weight network-specific node classifiers,
     a transferability weight learning module,
     a class-aware hierarchical graph contrastive domain adaptation module at both node and prototype levels,
     and a positive and negative pseudo-labeling module.


    Parameters:
    ----------
    num_layers: int
        The number of GNN layers.
    in_dim: int
        The input feature dimension.
    num_hidden: int
        The output feature dimension of the last GNN layer.
    heads: list of int
        The attention heads in each GNN layer.
    activation: callable activation function/layer or None, optional
        If not None, then apply an activation function to the updated node features.
        Default: ``None``.
    feat_drop: float
        The dropout rate on the input node features.
    attn_drop: float
        The dropout rate on the attention weights.
    negative_slope: float
        The negative slope parameter for the LeakyReLU function.
    residual: bool
        If True, use residual connection. Default: ``True``.
    num_classes: int
        The number of classes for node classification.
    num_source: int
        The number of source domains.
    """

    def __init__(self, num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope,
                 residual, num_classes, num_source):
        super(MHGCDA, self).__init__()
        self.network_embedding = GAT(num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop,
                                     negative_slope)
        self.node_classifier_list = NodeClassifier(num_hidden, heads, feat_drop, attn_drop, negative_slope, residual,
                                                   num_classes, num_source)

    def forward(self, num_source, features_s_list, features_t, g_s_list, g_t):
        h_s_list = [self.network_embedding(features_s_list[i], g_s_list[i]) for i in range(num_source)]
        pred_logit_s_list = self.node_classifier_list(h_s_list, g_s_list)
        emb_s_list = [h_s_list[i].reshape(h_s_list[i].shape[0], -1) for i in range(num_source)]
        h_t = self.network_embedding(features_t, g_t)
        pred_logit_t_list = self.node_classifier_list(h_t, g_t, is_target=True)
        emb_t = h_t.reshape(h_t.shape[0], -1)
        return pred_logit_s_list, pred_logit_t_list, emb_s_list, emb_t