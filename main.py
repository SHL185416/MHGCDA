import argparse
import random
from torch import nn
from model import MHGCDA
import dgl
from utils import *
from modules import *
from metrics import f1_scores

# 0. Set model parameters
parser = argparse.ArgumentParser()

# 0.1 Equipment and number of iterations settings
parser.add_argument("--gpu", type=int, default=0, help="which GPU to use. Set -1 to use CPU.")
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument("--batch_size", type=int, default=5000, help="batch_size for each domain")
# 0.2 Optimizer settings
parser.add_argument('--lr_ini', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--l2_w', type=float, default=0.01, help='weight of L2-norm regularization')
# 0.3 GAT settings
parser.add_argument("--num_heads", type=int, default=16, help="number of hidden attention heads")
parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
parser.add_argument("--num_hidden", type=int, default=16, help="number of hidden units")
parser.add_argument("--num_out_heads", type=int, default=3, help="number of output attention heads")
parser.add_argument("--in_drop", type=float, default=0.3, help="input feature dropout")
parser.add_argument("--attn_drop", type=float, default=0.7, help="attention dropout")
parser.add_argument("--random_number", type=int, default=1, help="random seed")
# 0.4 Contrastive domain adaptation settings
parser.add_argument("--tau", type=float, default=1, help="temperature-scales")
# 0.5 Pseudo label learning settings
parser.add_argument("--tau_p_rate", type=float, default=0.8, help="tau_n_rate for pseudo label learning")
parser.add_argument("--tau_n_rate", type=float, default=0.2, help="tau_p_rate for pseudo label learning")
# 0.6 The trade-off parameters
parser.add_argument("--Clf_wei", type=float, default=1, help="weight of clf loss")
parser.add_argument("--P_wei", type=float, default=1, help="weight of pseudo label learning loss")
parser.add_argument("--PP_wei", type=float, default=1, help="weight of PP-GCDA loss")
parser.add_argument("--NN_wei", type=float, default=1, help="weight of NN-GCDA loss")
# 0.7 Dataset settings
parser.add_argument('--target', type=str, default='citation1_citationv1', help='target dataset name')
parser.add_argument('--data_key', type=str, default='citation', help='dataset key')
args = parser.parse_args()
# 1. Set device and file path
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'
f = open(f'./output/{args.data_key}/otherSet_{args.target}.txt', 'a')
f.write('{}\n'.format(args))
f.flush()

source_list = ["citation1_acmv9", "citation1_citationv1", "citation1_dblpv7"]
if args.target not in source_list:
    source_list = ["citation2_acmv8", "citation2_citationv1", "citation2_dblpv4"]
source_list.remove(args.target)
num_source = len(source_list)
A_s_list = [0] * num_source
X_s_list = [0] * num_source
Y_s_list = [0] * num_source
num_nodes_s_list = [0] * num_source
g_s_list = [0] * num_source
# 2. Load dataset
# 2.1 Load data from the source network
for i in range(num_source):
    A_s_list[i], X_s_list[i], Y_s_list[i] = load_citation(f"./input/{args.data_key}/{source_list[i]}.mat")
    num_nodes_s_list[i] = X_s_list[i].shape[0]
    g_s_list[i] = dgl.from_scipy(A_s_list[i]).to(args.device).remove_self_loop().add_self_loop()
num_feat = X_s_list[0].shape[1]
num_class = Y_s_list[0].shape[1]
# 2.2 Load data from the target network
A_t, X_t, Y_t = load_citation(f"./input/{args.data_key}/{args.target}.mat")
num_nodes_t = X_t.shape[0]
g_t = dgl.from_scipy(A_t).to(args.device).remove_self_loop().add_self_loop()
features_s_list = [torch.Tensor(X_s_list[i].todense()).to(args.device) for i in range(num_source)]
features_t = torch.Tensor(X_t.todense()).to(args.device)
Y_s_tensor_list = [torch.LongTensor(Y_s_list[i]).to(args.device) for i in range(num_source)]
# 3. Definitions of model variables
heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
ST_max = max([X_s_list[i].shape[0] for i in range(0, num_source)])
ST_max = max(ST_max, X_t.shape[0])
microAllRandom = []
macroAllRandom = []
best_microAllRandom = []
best_macroAllRandom = []
numRandom = args.random_number + 5
tau_n_rate = args.tau_n_rate
tau_p_rate = args.tau_p_rate

for random_state in range(args.random_number, numRandom):
    print('%d-th random split' % random_state)
    # 4. Set the nonce seed, model and loss
    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state) if torch.cuda.is_available() else None
    np.random.seed(random_state)
    clf_type = 'multi-class'
    model = MHGCDA(
        num_layers=args.num_layers,
        in_dim=num_feat,
        num_hidden=args.num_hidden,
        heads=heads,
        activation=F.elu,
        feat_drop=args.in_drop,
        attn_drop=args.attn_drop,
        negative_slope=0.2,
        residual=False,
        num_classes=num_class,
        num_source=num_source
    )
    model = model.to(args.device)
    clf_loss_f = nn.BCEWithLogitsLoss(reduction='none') if clf_type == 'multi-label' \
        else nn.CrossEntropyLoss()
    best_epoch = 0
    best_micro_f1 = 0
    best_macro_f1 = 0
    pred_label = np.zeros(Y_t.shape)

    for epoch in range(1, args.epochs + 1):
        # 5. Use random sampling to sample from a dataset
        batch_s_list = [0] * num_source
        args_list = [mini_batch(X_s_list[i], Y_s_list[i], A_s_list[i], ST_max, args.batch_size) for i in
                     range(num_source)]
        args_list.append(mini_batch(X_t, pred_label, A_t, ST_max, args.batch_size))
        for batch_idx, batch_data in enumerate(zip(*args_list)):
            batch_s_list = [data[:4] for data in batch_data[:-1]]
            batch_t = batch_data[-1]
            feat_s_list = [0] * num_source
            label_s_list = [0] * num_source
            adj_s_list = [0] * num_source
            shuffle_index_s_list = [0] * num_source
            g_s1_list = [0] * num_source
            for i in range(num_source):
                feat_s_list[i], label_s_list[i], adj_s_list[i], shuffle_index_s_list[i] = batch_s_list[i]
                feat_s_list[i] = torch.FloatTensor(feat_s_list[i].toarray()).to(args.device)
                label_s_list[i] = torch.FloatTensor(label_s_list[i]).to(args.device)
                adj_s_list[i] = torch.FloatTensor(adj_s_list[i].toarray()).to(args.device)
                g_s1_list[i] = dgl.from_scipy(sp.coo_matrix(adj_s_list[i].cpu())).to(args.device)
                g_s1_list[i] = dgl.remove_self_loop(g_s1_list[i])
                g_s1_list[i] = dgl.add_self_loop(g_s1_list[i])
            feat_t, pred_label_t, adj_t, shuffle_index_t = batch_t
            feat_t = torch.FloatTensor(feat_t.toarray()).to(args.device)
            pred_label_t = torch.FloatTensor(pred_label_t).to(args.device)
            adj_t = torch.FloatTensor(adj_t.toarray()).to(args.device)
            g_t1 = dgl.from_scipy(sp.coo_matrix(adj_t.cpu())).to(args.device).remove_self_loop().add_self_loop()

            p = float(epoch) / args.epochs
            lr = args.lr_ini / (1. + 10 * p) ** 0.75
            # 6. Train the model
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=args.l2_w)
            optimizer.zero_grad()
            pred_logit_s_list, pred_logit_t_list, emb_s_list, emb_t = model(num_source,
                                                                            feat_s_list,
                                                                            feat_t,
                                                                            g_s1_list,
                                                                            g_t1, )
            """6.1 transferability weights based on information entropy"""
            w_k_list = transferability_weights(num_source, pred_logit_t_list)

            pred_logit_t_withWK = 0
            for i in range(num_source):
                pred_logit_t_withWK = pred_logit_t_withWK + pred_logit_t_list[i].detach() * w_k_list[i]
            """6.2 node classification loss"""
            clf_loss_list = [0] * num_source
            if clf_type == 'multi-class':
                for i in range(num_source):
                    clf_loss_list[i] = w_k_list[i] * clf_loss_f(pred_logit_s_list[i],
                                                                torch.argmax((label_s_list[i]), 1))
            else:
                for i in range(num_source):
                    clf_loss_list[i] = clf_loss_f(pred_logit_s_list[i], (label_s_list[i]).float())
                    clf_loss_list[i] = w_k_list[i] * torch.sum(clf_loss_list[i]) / (label_s_list[i]).shape[0]

            """6.3 positive and negative pseudo-labeling"""
            loss_p = pseudo_labeling(pred_logit_t_list, tau_p_rate, tau_n_rate, w_k_list)

            """6.4 node-level graph contrastive loss"""
            node_level_graph_contrastive_loss_list = [0] * num_source
            for i in range(num_source):
                comlabel_inter_st = torch.mm(label_s_list[i].float(), pred_label_t.t())
                comlabel_inter_ts = torch.mm(pred_label_t, label_s_list[i].float().t())
                cross_view_st = node_level_graph_contrastive_loss(emb_s_list[i], emb_t, args.tau, comlabel_inter_st,
                                                                  pred_label_t, True)
                cross_view_ts = node_level_graph_contrastive_loss(emb_t, emb_s_list[i], args.tau, comlabel_inter_ts,
                                                                  pred_label_t, False)
                node_level_graph_contrastive_loss_list[i] = w_k_list[i] * 0.5 * (
                        cross_view_st.mean() + cross_view_ts.mean())
            """6.4 prototype-level graph contrastive loss"""
            prototype_level_graph_contrastive_loss_list = [0] * num_source
            label_t_sum = (pred_label_t.sum(0) > 0).float()
            O_TT = label_t_sum.view(-1, 1) * label_t_sum.view(1, -1)
            prototype_t = calculate_prototype(pred_label_t, emb_t)
            for i in range(num_source):
                label_s_sum = (label_s_list[i].sum(0) > 0).float()
                O_ST = label_s_sum.view(-1, 1) * label_t_sum.view(1, -1)
                O_SS = label_s_sum.view(-1, 1) * label_s_sum.view(1, -1)
                prototype_s = calculate_prototype(label_s_list[i], emb_s_list[i])
                cross_view_st = prototype_level_graph_contrastive_loss(prototype_s, prototype_t, args.tau, O_ST, O_SS)
                cross_view_ts = prototype_level_graph_contrastive_loss(prototype_t, prototype_s, args.tau, O_ST.T, O_TT)
                prototype_level_graph_contrastive_loss_list[i] = w_k_list[i] * 0.5 * (
                        cross_view_st.mean() + cross_view_ts.mean())

            total_loss = args.Clf_wei * sum(clf_loss_list) + \
                         args.PP_wei * sum(prototype_level_graph_contrastive_loss_list) + \
                         args.NN_wei * sum(node_level_graph_contrastive_loss_list) + \
                         args.P_wei * loss_p
            total_loss.backward()
            optimizer.step()

        '''7. Compute evaluation on test data by the end of each epoch'''
        model.eval()
        with torch.no_grad():
            pred_logit_s_list, pred_logit_t_list, emb_s_list, emb_t = model(num_source,
                                                                            features_s_list,
                                                                            features_t,
                                                                            g_s_list,
                                                                            g_t, )
            """7.1 transferability weights based on information entropy"""
            w_k_list = transferability_weights(num_source, pred_logit_t_list)
            print("eval W_k:", w_k_list)

            pred_logit_t_withWK = 0
            for i in range(num_source):
                pred_logit_t_withWK = pred_logit_t_withWK + pred_logit_t_list[i].detach() * w_k_list[i]
            """7.2 node classification loss"""
            clf_loss_list = [0] * num_source
            if clf_type == 'multi-class':
                for i in range(num_source):
                    clf_loss_list[i] = clf_loss_f(pred_logit_s_list[i], torch.argmax((Y_s_tensor_list[i]), 1))
            else:
                for i in range(num_source):
                    clf_loss_list[i] = clf_loss_f(pred_logit_s_list[i], (Y_s_tensor_list[i]).float())
                    clf_loss_list[i] = torch.sum(clf_loss_list[i]) / (Y_s_tensor_list[i]).shape[0]
            for i in range(num_source):
                print('clf_loss%d : %f' % (i, clf_loss_list[i]))
            # 7.3 Calculates the probabilities for the source and target domains
            pred_prob_xs_list = [0] * num_source
            for i in range(num_source):
                pred_prob_xs_list[i] = F.sigmoid(pred_logit_s_list[i]) if clf_type == 'multi-label' else F.softmax(
                    pred_logit_s_list[i], dim=1)
            pred_prob_xt = F.sigmoid(pred_logit_t_withWK) if clf_type == 'multi-label' else F.softmax(
                pred_logit_t_withWK, dim=1)
            # 7.4 Calculate s-t f1_scores
            for i in range(num_source):
                f1_s = f1_scores(pred_prob_xs_list[i].cpu(), Y_s_list[i])
                print('epoch %d: Source%d micro-F1: %f, macro-F1: %f' % (epoch, i, f1_s[0], f1_s[1]))
            f1_t = f1_scores(pred_prob_xt.cpu(), Y_t)
            print('epoch %d: Target testing micro-F1: %f, macro-F1: %f' % (epoch, f1_t[0], f1_t[1]))
            # 7.5 Calculate the classifier f1_scores
            pred_label_clf_list = [0] * num_source
            for i in range(num_source):
                _, indices = torch.max(pred_logit_t_list[i], dim=1)
                pred_label_clf_list[i] = one_hot_encode_torch(indices, pred_logit_t_list[i].shape[1])
                print("accuracy of clf%d label" % i, f1_scores(pred_label_clf_list[i], Y_t))
            # 7.6 Calculate Pseudo-labels
            pred_label = calculate_pred_label_t(pred_label_clf_list).float()
            if epoch % 5 == 0:
                print(f"{epoch + 1} pseudo_label {pred_label.sum() / pred_label.shape[0]} %")
            if torch.any(pred_label != 0, dim=1).sum() > 0:
                print("accuracy of refined label by both clustering and clf",
                      f1_scores(pred_label[torch.any(pred_label != 0, dim=1)], Y_t[torch.any(pred_label != 0, dim=1)]))

            if f1_t[1] > best_macro_f1:
                best_micro_f1 = f1_t[0]
                best_macro_f1 = f1_t[1]
                best_epoch = epoch

    print('Target best epoch %d, micro-F1: %f, macro-F1: %f' % (best_epoch, best_micro_f1, best_macro_f1))
    microAllRandom.append(float(f1_t[0]))
    macroAllRandom.append(float(f1_t[1]))
    best_microAllRandom.append(float(best_micro_f1))
    best_macroAllRandom.append(float(best_macro_f1))
# 8. Record the results
micro = np.mean(microAllRandom)
macro = np.mean(macroAllRandom)
micro_sd = np.std(microAllRandom)
macro_sd = np.std(macroAllRandom)
print("The avergae micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: ".format(
    numRandom - 1, micro, micro_sd, macro, macro_sd))
f.write("The avergae micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: \n".format(
    numRandom - 1, micro, micro_sd, macro, macro_sd))

best_micro = np.mean(best_microAllRandom)
best_macro = np.mean(best_macroAllRandom)
best_micro_sd = np.std(best_microAllRandom)
best_macro_sd = np.std(best_macroAllRandom)
print(
    "The avergae best micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: ".format(
        numRandom - 1, best_micro, best_micro_sd, best_macro, best_macro_sd))
f.write(
    "The avergae best micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: \n".format(
        numRandom - 1, best_micro, best_micro_sd, best_macro, best_macro_sd))

f.flush()
f.close()
