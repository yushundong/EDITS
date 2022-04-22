import time
import argparse
import numpy as np

import torch
print(torch.__version__)

import torch.nn.functional as F
import torch.optim as optim
from metrics import metric_wd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import scipy.sparse as sp
from utils import load_bail, load_credit, load_german, feature_norm, normalize_scipy
from gcn import GCN
from torch_geometric.utils import dropout_adj, convert
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

torch.cuda.set_device(0)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--cuda_device', type=int, default=0,
                    help='cuda device running on.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--preprocessed_using', type=int, default=0,
                    help='1 and 0 represent utilizing and not utilizing the preprocessed results.')
parser.add_argument('--dataset', type=str, default='bail',
                    help='a dataset from credit, german and bail.')
parser.add_argument('--epochs', type=int, default=1000,  # german: 1300 stability purpose
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)



if args.dataset == 'credit':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit('credit', label_number=6000)
elif args.dataset == 'german':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_german('german', label_number=100)
elif args.dataset == 'bail':
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail('bail', label_number=100)
else:
    print("This dataset is not supported up to now!")

adj_ori = adj
adj = normalize_scipy(adj)


if args.preprocessed_using:
    A_debiased, features = sp.load_npz('pre_processed/A_debiased.npz'), torch.load("pre_processed/X_debiased.pt", map_location=torch.device('cpu')).cpu().float()
    threshold_proportion = 0.015  # GCN: {credit: 0.02, german: 0.29, bail: 0.015}
    the_con1 = (A_debiased - adj_ori).A
    the_con1 = np.where(the_con1 > np.max(the_con1) * threshold_proportion, 1 + the_con1 * 0, the_con1)
    the_con1 = np.where(the_con1 < np.min(the_con1) * threshold_proportion, -1 + the_con1 * 0, the_con1)
    the_con1 = np.where(np.abs(the_con1) == 1, the_con1, the_con1 * 0)
    A_debiased = adj_ori + sp.coo_matrix(the_con1)
    assert A_debiased.max() == 1
    assert A_debiased.min() == 0
    features = features[:, torch.nonzero(features.sum(axis=0)).squeeze()].detach()
    A_debiased = normalize_scipy(A_debiased)

if args.dataset != 'german':
    preserve = features
    features = feature_norm(features)
    if args.preprocessed_using == 0 and args.dataset == 'credit':
        features[:, 1] = preserve[:, 1]  # credit
    elif args.preprocessed_using == 0 and args.dataset == 'bail':
        features[:, 0] = preserve[:, 0]  # bail

if args.preprocessed_using:
    print("****************************After debiasing****************************")
    metric_wd(features, A_debiased, sens, 0.9, 0)
    metric_wd(features, A_debiased, sens, 0.9, 2)
    print("****************************************************************************")
    X_debiased = features.float()
    edge_index = convert.from_scipy_sparse_matrix(A_debiased)[0].cuda()
else:
    print("****************************Before debiasing****************************")
    metric_wd(features, adj, sens, 0.9, 0)
    metric_wd(features, adj, sens, 0.9, 2)
    print("****************************************************************************")
    X_debiased = features.float()
    edge_index = convert.from_scipy_sparse_matrix(adj)[0].cuda()


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

# Model and optimizer
model = GCN(nfeat=X_debiased.shape[1], nhid=args.hidden, nclass=labels.max().item(), dropout=args.dropout).float()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    torch.cuda.set_device(args.cuda_device)
    model.cuda()
    X_debiased = X_debiased.cuda()
    labels = labels.cuda()
    idx_train = idx_train
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch, pa, eq, test_f1, val_loss, test_auc):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
    preds = (output.squeeze() > 0).type_as(labels)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
    auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train.cpu().numpy()], output.detach().cpu().numpy()[idx_train.cpu().numpy()])
    f1_train = f1_score(labels[idx_train.cpu().numpy()].cpu().numpy(), preds[idx_train.cpu().numpy()].cpu().numpy())
    loss_train.backward()
    optimizer.step()
    _, _ = fair_metric(preds[idx_train.cpu().numpy()].cpu().numpy(), labels[idx_train.cpu().numpy()].cpu().numpy(), sens[idx_train.cpu().numpy()].cpu().numpy())

    model.eval()
    output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
    preds = (output.squeeze() > 0).type_as(labels)
    loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
    auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val.cpu().numpy()], output.detach().cpu().numpy()[idx_val.cpu().numpy()])
    f1_val = f1_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())
    # print('Epoch: {:04d}'.format(epoch + 1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'F1_train: {:.4f}'.format(f1_train),
    #       'AUC_train: {:.4f}'.format(auc_roc_train),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'F1_val: {:.4f}'.format(f1_val),
    #       'AUC_val: {:.4f}'.format(auc_roc_val),
    #       'time: {:.4f}s'.format(time.time() - t))

    if epoch < 15:
        return 0, 0, 0, 1e5, 0
    if loss_val < val_loss:
        val_loss = loss_val.data
        pa, eq, test_f1, test_auc = test(test_f1)
        # print("Parity of val: " + str(pa))
        # print("Equality of val: " + str(eq))
    return pa, eq, test_f1, val_loss, test_auc


def test(test_f1):
    model.eval()
    output = model(x=X_debiased, edge_index=torch.LongTensor(edge_index.cpu()).cuda())
    preds = (output.squeeze() > 0).type_as(labels)
    loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
    auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu().numpy()], output.detach().cpu().numpy()[idx_test.cpu().numpy()])
    f1_test = f1_score(labels[idx_test.cpu().numpy()].cpu().numpy(), preds[idx_test.cpu().numpy()].cpu().numpy())
    test_auc = auc_roc_test
    test_f1 = f1_test
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "F1_test= {:.4f}".format(test_f1),
    #       "AUC_test= {:.4f}".format(test_auc))
    parity_test, equality_test = fair_metric(preds[idx_test.cpu().numpy()].cpu().numpy(),
                                               labels[idx_test.cpu().numpy()].cpu().numpy(),
                                               sens[idx_test.cpu().numpy()].cpu().numpy())
    # print("Parity of test: " + str(parity_test))
    # print("Equality of test: " + str(equality_test))
    return parity_test, equality_test, test_f1, test_auc


# Train model
t_total = time.time()
val_loss = 1e5
pa = 0
eq = 0
test_auc = 0
test_f1 = 0
for epoch in tqdm(range(args.epochs)):
    pa, eq, test_f1, val_loss, test_auc = train(epoch, pa, eq, test_f1, val_loss, test_auc)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print("Delta_{SP}: " + str(pa))
print("Delta_{EO}: " + str(eq))
print("F1: " + str(test_f1))
print("AUC: " + str(test_auc))
