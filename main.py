import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.cuda import amp
from utils.dataloader import SocLoader
from model_hetero.Simple_HGN import *
from model_hetero.LightGCN import *
from model_hetero.TailSTEAK import TailSTEAK

from sklearn.manifold import TSNE
import seaborn as sns
import argparse, random

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='running mode, choose between [train, test]')
parser.add_argument('--model', '-m', type=str, default='GCN', help='name of models')
parser.add_argument('--debias_on', '-do', type=int, default=1, help='Use Tail-STEAK or not. If not, set debias_on=0')
parser.add_argument('--dataset', '-d', type=str, default='deezer', help='chosen dataset')
parser.add_argument('--init_node_feat', '-nf', type=str, default='', help='path of node feature')
parser.add_argument('--graph_path', '-g', type=str, default='', help='path of graph structure')
parser.add_argument('--dataset_path', '-dp', type=str, default='', help='path of dataset-specific directory')
parser.add_argument('--checkpoint_dir', '-cd', type=str, default='checkpoint', help='path of checkpoint file')
parser.add_argument('--output_dir', '-od', default='output', help='directory path for output degree-related data')
parser.add_argument('--num_etype', type=int, default=1, help='number of relation type')
parser.add_argument('--edge_channel', type=int, default=32, help='dimension of relation embedding if involved')
parser.add_argument('--gamma', type=int, default=1, help='maximum of kept neighbors')
parser.add_argument('--U', type=int, default=500, help='Size of candidate strangers for pseudo link prediction')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id chosen to run train/test')
parser.add_argument('--layer_num', type=int, default=1, help='number of stacked model layer')
parser.add_argument('--input_dim', type=int, default=32, help='dimension of input feature')
parser.add_argument('--hidden_channel', type=int, default=32, help='dimension of hidden layer')
parser.add_argument('--weight_decay', type=float, default=0, help='hyperparameter of weight decay')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--neg_sample_num', type=int, default=19, help='number of negative samples during training and validation')
parser.add_argument('--neg_sample_num_test', type=int, default=99, help='number of negative samples during testing')
parser.add_argument('--eval_per_n', type=int, default=1000, help='evaluate per n steps')
parser.add_argument('--epoch_num', type=int, default=30, help='training epoch number')
parser.add_argument('--ft_epoch', type=int, default=5, help='Epoch number for second stage')
parser.add_argument('--ssl_temp', type=float, default=0.5, help='temperature hyperparameter for all contrastive learning based methods')
parser.add_argument('--ssl_reg', type=float, default=1.0, help='coefficient for contrastive learning loss')
parser.add_argument('--deg_t_low', type=float, default=15, help='degree threshold set for our GCL method')

args, _ = parser.parse_known_args()

ssl_on = args.debias_on == 1
base_model = args.model

# Print current experimental setting
print("Run %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
print("Current process ID: %d" % os.getpid())
print("Current Dataset: %s" % args.dataset)
print("Current Model: %s" % args.model)
print("Running Mode: %s" % args.mode)
print("------HyperParameter Settings------")
print("Number of Stacking Layer: %d" % args.layer_num)
print("In Channel: %d" % args.input_dim)
print("Hidden Channel: %d" % args.hidden_channel)
if args.mode == 'train':
    print("Learning Rate: %f" % args.lr)
    print("Weight Decay: %f" % args.weight_decay)
    print("Batch Size: %d" % args.batch_size)
    print("Epoch Number: %d" % args.epoch_num)
    print("Evaluate per %d step" % args.eval_per_n)
    if ssl_on:
        print("Temperature for SSL: %.2f" % args.ssl_temp)
        print("SSL Loss Coefficient: %.2f" % args.ssl_reg)
        print("Degree Threshold: %d" % args.deg_t_low)
        print("Gamma: %d" % args.gamma)
        print("U: %d" % args.U)
        print("Second Stage Epoch Number: %d" % args.ft_epoch)
    print("\n\n")
else:
    print("Batch Size: %d" % args.batch_size)
    print("\n\n")

node_feat = torch.tensor(np.load(args.init_node_feat)).float()
print(node_feat.size(1))
# if args.mode == 'train':
if ssl_on:
    train_loader = SocLoader(node_feat, batch_size=args.batch_size, mode='train', shuffle=True, debias_on=ssl_on, gamma=args.gamma, deg_t_low=args.deg_t_low, graph_path=args.graph_path)
else:
    train_loader = SocLoader(node_feat, batch_size=args.batch_size, mode='train', shuffle=True, graph_path=args.graph_path)
val_loader = SocLoader(node_feat, batch_size=args.batch_size, mode='val', shuffle=False, graph_path=args.graph_path)
test_loader = SocLoader(node_feat, batch_size=args.batch_size, mode='test', shuffle=True, graph_path=args.graph_path)
print("base graph loaded")

if args.gpu_id != -1:
    device = torch.device('cuda:%d' % args.gpu_id if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
enable_amp = True if "cuda" in device.type else False

if ssl_on:
    model = TailSTEAK(args.input_dim, args.hidden_channel, node_feat, args.num_etype, args.edge_channel, args.layer_num, 
                      ssl_temp=args.ssl_temp, deg_t_low=args.deg_t_low, U=args.U, base=base_model)
elif args.model == 'HGN':
    model = HGN(args.input_dim, args.hidden_channel, node_feat, args.num_etype, args.edge_channel, args.layer_num)
elif args.model == 'LightGCN':
    model = LightGCN(args.input_dim, args.layer_num, node_feat)
else:
    print("Model is not pre-defined!")
    exit(0)


model = model.to(device)
node_feat = node_feat.to(device)
if args.mode == 'train':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler(enabled=enable_amp)


# ====================== Evaluation Metrics =========================
def ndcg_score(y_true, y_score, k=10):
    y_standard = np.zeros_like(y_true)
    for i in range(y_standard.shape[1]):
        y_standard[:, i] = i
    best = dcg_score(y_true, y_standard, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def dcg_score(y_true, y_pred, k=10):
    y_true = np.take(y_true, y_pred[:, :k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(np.shape(y_true)[1]) + 2)
    return np.sum(gains / discounts, axis=1)


def hit_rate(logits, k=10):
    sorted_idx = logits[:, :k]
    size = logits.shape[0]
    tot = 0.
    for i in range(size):
        if 0 in sorted_idx[i, :]:
            tot += 1
    return tot / size


# =========================== Train Code =============================
def train(best_ndcg, x, full_edge_index, cur_epoch):
    model.train()
    step = 0
    out_deg, in_deg = train_loader.homo_out_deg.to(device), train_loader.homo_in_deg.to(device)
    for out in train_loader:
        if not ssl_on:
            _, x, adjs, node_idx, n_id, adjs_2, node_idx_2, n_id_2, out_deg, in_deg = out 
        else:
            batch_len, x, adjs, node_idx, n_id, ssl_x, ssl_adjs, ssl_node_idx, ssl_n_id, \
                diff_x, diff_adjs, diff_node_idx, diff_n_id, out_deg, in_deg = out
            ssl_x = ssl_x.to(device)
            ssl_adjs = [adj.to(device) for adj in ssl_adjs]
            if diff_x is not None:
                diff_x = diff_x.to(device)
                diff_adjs = [adj.to(device) for adj in diff_adjs]
                diff_node_idx = diff_node_idx.to(device)
            ssl_node_idx = ssl_node_idx.to(device)
        
        if ssl_on or args.model not in ['LightGCN', 'SAGE', 'GCN', 'GIN']:
            edge_index = full_edge_index
            for i in range(len(edge_index)):
                edge_index[i] = [arr.to(device) for arr in edge_index[i]]
        else:
            edge_index = full_edge_index[0]
            edge_index = [arr.to(device) for arr in edge_index]
        in_deg, out_deg = in_deg.to(device), out_deg.to(device)
        n_id = n_id.to(device)

        optimizer.zero_grad()
        with amp.autocast(enabled=enable_amp):
            assert len(n_id) % (2 + args.neg_sample_num) == 0
            if ssl_on:
                pos_logit, neg_logits, ssl_loss_1 = model(x, edge_index, n_id, ssl_x=ssl_x, ssl_adjs=ssl_adjs, ssl_n_id=ssl_n_id, 
                                                                    neg_sample_num=args.neg_sample_num, out_deg=out_deg, node_idx=node_idx, ssl_node_idx=ssl_node_idx, device=device)
            else:
                pos_logit, neg_logits = model(x, edge_index, n_id, args.neg_sample_num, node_idx=node_idx, out_deg=out_deg)
        
            logits = torch.cat([pos_logit.reshape(1, -1), neg_logits.reshape(args.neg_sample_num, -1)], dim=0).T
            pos_loss = F.logsigmoid(pos_logit).mean()
            neg_loss = F.logsigmoid(-neg_logits).reshape(args.neg_sample_num, -1).mean(dim=0).mean()
            loss = -pos_loss - neg_loss
            if ssl_on:
                loss = loss + ssl_loss_1 * args.ssl_reg

        if np.isnan(loss.detach().cpu()):
            print("NaN Loss Detected")
            return
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if step % 20 == 0: 
            logits = logits.detach().cpu()
            sorted_idx = torch.argsort(logits, dim=-1, descending=True).numpy()
            hr_1 = hit_rate(sorted_idx, k=1)
            hr_5 = hit_rate(sorted_idx, k=5)
            hr_10 = hit_rate(sorted_idx, k=10)
            print("step %d : loss %.4f, HR@1 %.4f, HR@5 %.4f, HR@10 %.4f" % (step, float(loss), hr_1, hr_5, hr_10))
        if step != 0 and step % args.eval_per_n == 0:
            start = time.time()
            hr_1, hr_5, hr_10, mrr, ndcg = test('val')
            end = time.time()
            print("time consumption: %.4f" % (end-start))
            print("Test result on validation set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG@10: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                print("New best model saved")
                torch.save(model.state_dict(), checkpoint_path_pt)
        step += 1

    start = time.time()
    hr_1, hr_5, hr_10, mrr, ndcg = test('val')
    end = time.time()
    print("time consumption: %.4f" % (end - start))
    print("Test result on validation set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        torch.save(model.state_dict(), checkpoint_path_pt)
        print("New best model saved")
    
    return best_ndcg


def decoder_finetune(best_ndcg, x, edge_index):
    model.train()
    step = 0
    out_deg, in_deg = train_loader.homo_out_deg.to(device), train_loader.homo_in_deg.to(device)
    for out in train_loader:
        if not ssl_on:
            batch_len, x, adjs, node_idx, n_id, out_deg, in_deg = out
        else:
            batch_len, x, adjs, node_idx, n_id, ssl_x, ssl_adjs, ssl_node_idx, ssl_n_id, \
                diff_x, diff_adjs, diff_node_idx, diff_n_id, out_deg, in_deg = out
            ssl_x = ssl_x.to(device)
            ssl_adjs = [adj.to(device) for adj in ssl_adjs]
            if diff_x is not None:
                diff_x = diff_x.to(device)
                diff_adjs = [adj.to(device) for adj in diff_adjs]
                diff_node_idx = diff_node_idx.to(device)
            ssl_node_idx = ssl_node_idx.to(device)
        
        x = x.to(device)
        diff_eindex = []
        extra_eindex = train_loader.pseudo_edge_index.to(device)
        for i in range(len(edge_index)):
            edge_index[i] = [arr.to(device) for arr in edge_index[i]]
            eindex = edge_index[i][0]
            diff_eindex.append([torch.cat([eindex, extra_eindex], dim=-1), edge_index[i][1]])

        in_deg, out_deg = in_deg.to(device), out_deg.to(device)
        n_id = n_id.to(device)

        optimizer.zero_grad()
        with amp.autocast(enabled=enable_amp):
            assert len(n_id) % (2 + args.neg_sample_num) == 0
            if ssl_on:
                pos_logit, neg_logits, ssl_loss_1 = model(x, edge_index, n_id, ssl_x=ssl_x, ssl_adjs=ssl_adjs, ssl_n_id=ssl_n_id, diff_adjs=diff_eindex,
                                                        neg_sample_num=args.neg_sample_num, out_deg=out_deg, node_idx=node_idx, ssl_node_idx=ssl_node_idx, mode='ft', device=device)
            else:
                pos_logit, neg_logits = model(x, edge_index, n_id, args.neg_sample_num, node_idx=node_idx, out_deg=out_deg)
            
            logits = torch.cat([pos_logit.reshape(1, -1), neg_logits.reshape(args.neg_sample_num, -1)], dim=0).T
            pos_loss = F.logsigmoid(pos_logit).mean()
            neg_loss = F.logsigmoid(-neg_logits).reshape(args.neg_sample_num, -1).mean(dim=0).mean()
            loss = -pos_loss - neg_loss + ssl_loss_1 * args.ssl_reg

        if np.isnan(loss.detach().cpu()):
            print("NaN Loss Detected")
            return
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if step % 20 == 0: 
            logits = logits.detach().cpu()
            sorted_idx = torch.argsort(logits, dim=-1, descending=True).numpy()
            hr_1 = hit_rate(sorted_idx, k=1)
            hr_5 = hit_rate(sorted_idx, k=5)
            hr_10 = hit_rate(sorted_idx, k=10)
            print("step %d : loss %.4f, HR@1 %.4f, HR@5 %.4f, HR@10 %.4f" % (step, float(loss), hr_1, hr_5, hr_10))
        if step != 0 and step % args.eval_per_n == 0:
            start = time.time()
            hr_1, hr_5, hr_10, mrr, ndcg = test('val')
            end = time.time()
            print("time consumption: %.4f" % (end-start))
            print("Test result on validation set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG@10: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                print("New best model saved")
                torch.save(model.state_dict(), checkpoint_path)
        step += 1

    start = time.time()
    hr_1, hr_5, hr_10, mrr, ndcg = test('val')
    end = time.time()
    print("time consumption: %.4f" % (end - start))
    print("Test result on validation set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        print("New best model saved")
        torch.save(model.state_dict(), checkpoint_path)
    
    return best_ndcg


@torch.no_grad()
def test(mode):
    model.eval()
    step = 0
    hr_1_tot, hr_5_tot, hr_10_tot, mrr, ndcg_tot = 0, 0, 0, 0, 0
    edge_index = []
    if mode == 'val':
        n_id = val_loader.dataset.t().reshape(-1)
        if ssl_on or args.model not in ['LightGCN', 'SAGE', 'GCN', 'GIN']:
            for elist, wlist in zip(val_loader.edge_index, val_loader.edge_weight):
                edge_index.append([elist.to(device), wlist.to(device)])
        else:
            edge_index, edge_weight = val_loader.edge_index[0].to(device), val_loader.edge_weight[0].to(device)
            edge_index = [edge_index, edge_weight]
    else:
        if ssl_on:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(checkpoint_path_pt, map_location=torch.device('cpu')))
        n_id = test_loader.dataset.t().reshape(-1)
        if ssl_on or args.model not in ['LightGCN', 'SAGE', 'GCN', 'GIN']:
            for elist, wlist in zip(test_loader.edge_index, test_loader.edge_weight):
                edge_index.append([elist.to(device), wlist.to(device)])
        else:
            edge_index, edge_weight = test_loader.edge_index[0].to(device), test_loader.edge_weight[0].to(device)
            edge_index = [edge_index, edge_weight]
    
    print("Start test...")
    deg_hr = {}
    deg_mrr = {}
    x = node_feat.to(device)
    
    if mode == 'test':
        out_deg = test_loader.homo_out_deg[n_id]
        pos_logit, neg_logits = model(x, edge_index, n_id, args.neg_sample_num_test, out_deg=out_deg, mode='test')
        logits = torch.cat([pos_logit.reshape(1, -1), neg_logits.reshape(args.neg_sample_num_test, -1)], dim=0).T
        gt_deg = out_deg.reshape(args.neg_sample_num_test+2, -1).t().cpu().numpy()[:, 1:]
    else:
        out_deg = val_loader.homo_out_deg[n_id]
        pos_logit, neg_logits = model(x, edge_index, n_id, args.neg_sample_num, out_deg=out_deg, mode='test')
        logits = torch.cat([pos_logit.reshape(1, -1), neg_logits.reshape(args.neg_sample_num, -1)], dim=0).T

    sorted_idx = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
    if mode == 'test':
        pred_deg = []
        for idx in range(sorted_idx.shape[0]):
            pred_deg.append(gt_deg[idx, sorted_idx[idx, 0]])
        gt_deg = gt_deg[:, 0].tolist()
    hr_1 = hit_rate(sorted_idx, k=1)
    hr_5 = hit_rate(sorted_idx, k=5)
    hr_10 = hit_rate(sorted_idx, k=10)
    hr_1_tot += hr_1
    hr_5_tot += hr_5
    hr_10_tot += hr_10
    
    y_true = np.zeros_like(sorted_idx)
    y_true[:, 0] = 1
    
    y_true = np.take(y_true, sorted_idx)
    rr_score = y_true / (np.arange(np.shape(y_true)[1]) + 1)
    mrr_arr = np.sum(rr_score, axis=1) / np.sum(y_true, axis=1)
    mrr += np.mean(mrr_arr)

    y_true = np.zeros_like(sorted_idx)
    y_true[:, 0] = 1
    ndcg = ndcg_score(y_true, sorted_idx)
    ndcg_tot += np.mean(ndcg)

    if mode == 'test':
        gt_degs, pred_degs = [], []
        test_dataset = test_loader.dataset
        out_deg, in_deg = test_loader.homo_out_deg, test_loader.homo_in_deg
        out_deg = out_deg.cpu().numpy()[test_dataset.numpy()[:, 0]]
        in_deg = in_deg.cpu().numpy()[test_dataset.numpy()[:, 0]]
        for i in range(out_deg.shape[0]):
            if out_deg[i] not in deg_hr:
                deg_hr[out_deg[i]] = [ndcg[i]]
                deg_mrr[out_deg[i]] = [mrr_arr[i]]
            else:
                deg_hr[out_deg[i]].append(ndcg[i])
                deg_mrr[out_deg[i]].append(mrr_arr[i])
            if out_deg[i] == 0:
                gt_degs.append(gt_deg[i])
                pred_degs.append(pred_deg[i])
        gt_degs, pred_degs = Counter(gt_degs), Counter(pred_degs)

    step += 1
    if mode == 'test':
        return hr_1_tot / step, hr_5_tot / step, hr_10_tot / step, mrr / step, ndcg_tot / step,  (deg_hr, deg_mrr)
    else:
        return hr_1_tot / step, hr_5_tot / step, hr_10_tot / step, mrr / step, ndcg_tot / step


@torch.no_grad()
def pseudo_edge_predict(index=0):
    model.eval()
    edge_index = []
    n_id = torch.arange(train_loader.homo_out_deg.size(0), dtype=torch.int64)
    n_id = n_id[train_loader.homo_out_deg <= args.deg_t_low].to(device)
    print(n_id.size())
    for elist, wlist in zip(train_loader.edge_index, train_loader.edge_weight):
            edge_index.append([elist.to(device), wlist.to(device)])
    print("Start pseudo construction...")
    x = node_feat.to(device)
    
    out_deg = train_loader.homo_out_deg[n_id]
    pseudo_edge_index = []
    num_range = out_deg.size(0) // 2048 + 1
    for i in range(num_range):
        nid_batch = n_id[2048*i:min(out_deg.size(0), 2048*(i+1))]
        out_deg_batch = out_deg[2048*i:min(out_deg.size(0), 2048*(i+1))]
        sampled_idx, logits = model(x, edge_index, nid_batch, args.neg_sample_num, out_deg=out_deg_batch, mode='pseudo')
        logits = logits.reshape(nid_batch.size(0), args.U)   
        
        logits = logits.cpu()
        random_noise = torch.rand(logits.numpy().shape) * 1e-7
        logits = logits + random_noise

        sorted_idx = torch.argsort(logits, dim=-1, descending=True).numpy()
        sorted_idx = sorted_idx[:, :int(args.deg_t_low)]   
        
        nid_batch = nid_batch.cpu().numpy()
        for m, idx in enumerate(nid_batch):
            for n, tar in enumerate(sorted_idx[m]):
                true_tar = sampled_idx[m, tar]
                if idx != true_tar:
                    pseudo_edge_index.append([idx, true_tar])
    pseudo_edge_index = torch.tensor(pseudo_edge_index, dtype=torch.int64)
    return pseudo_edge_index


seed_set = [1000, 2307, 3407, 4508, 5510]
full_hr_1, full_hr_5, full_hr_10, full_mrr, full_ndcg = [], [], [], [], []
full_head_ndcg, full_tail_ndcg = [], []
full_head_mrr, full_tail_mrr = [], []
model_str = args.model + "_TS" if ssl_on else args.model
for num, seed_val in enumerate(seed_set):
    print("Run %d" % num)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    model.reset_parameters(node_feat)
    checkpoint_path_pt = '%s/%s_no_ft_%d.pth' % (args.checkpoint_dir, model_str, num)
    checkpoint_path = '%s/%s_%d.pth' % (args.checkpoint_dir, model_str, num)

    ndcg = 0
    x = node_feat
    edge_index = []
    for elist, wlist in zip(train_loader.edge_index, train_loader.edge_weight):
        edge_index.append([elist.to(device), wlist.to(device)])
    for epoch in range(1, args.epoch_num + 1):
        print("Currently epoch %d:" % epoch)
        ndcg = train(ndcg, x, edge_index, epoch)
    if ssl_on:
        model.load_state_dict(torch.load(checkpoint_path_pt))
        print("Start Decoder Fine-tune")
        ndcg = 0
        for epoch in range(1, args.ft_epoch):
            print("Currently epoch %d:" % epoch)
            pseudo_edge_index = pseudo_edge_predict(epoch-1)
            del train_loader
            train_loader = SocLoader(node_feat, batch_size=args.batch_size, mode='train', shuffle=True, debias_on=ssl_on, 
                                     gamma=args.gamma, deg_t_low=args.deg_t_low, graph_path=args.graph_path, pseudo_edge_index=pseudo_edge_index)
            ndcg = decoder_finetune(ndcg, x, edge_index)

    test_mrrs, test_hr = [], []
    degs, deg_nums, deg_hrs = [], [], []
    deg_mrrs = []
    start = time.time()
    hr_1, hr_5, hr_10, mrr, ndcg, (deg_hr, deg_mrr) = test('test')
    full_hr_1.append(hr_1)
    full_hr_5.append(hr_5)
    full_hr_10.append(hr_10)
    full_mrr.append(mrr)
    full_ndcg.append(ndcg)
    end = time.time()
    print("time consumption: %.4f" % (end-start))
    print("Test result on test set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG@10: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
    for key in deg_hr:
        degs.append(key)
        deg_nums.append(len(deg_hr[key]))
        deg_hrs.append(np.mean(deg_hr[key]))
        deg_mrrs.append(np.mean(deg_mrr[key]))

    np.save(args.output_dir + '/%s_deg_%d.npy' % (model_str, num), degs)
    np.save(args.output_dir + '/%s_deg_num_%d.npy' % (model_str, num), deg_nums)
    np.save(args.output_dir + '/%s_deg_ndcg_%d.npy' % (model_str, num), deg_hrs)

    degs, deg_nums, deg_hrs = np.array(degs), np.array(deg_nums), np.array(deg_hrs)
    deg_mrrs = np.array(deg_mrrs)
    idx = degs <= args.deg_t_low
    idx_head = degs > args.deg_t_low
    tail_ndcg = np.sum(deg_nums[idx] * deg_hrs[idx]) / np.sum(deg_nums[idx])
    head_ndcg = np.sum(deg_nums[idx_head] * deg_hrs[idx_head]) / np.sum(deg_nums[idx_head])
    tail_mrr = np.sum(deg_nums[idx] * deg_mrrs[idx]) / np.sum(deg_nums[idx])
    head_mrr = np.sum(deg_nums[idx_head] * deg_mrrs[idx_head]) / np.sum(deg_nums[idx_head])
    full_head_ndcg.append(head_ndcg)
    full_tail_ndcg.append(tail_ndcg)
    full_head_mrr.append(head_mrr)
    full_tail_mrr.append(tail_mrr)

print("\n")
print("Performance statistics are presented in two-row manner. The first row is mean val, the second row is std.")
print("Overall performance of " + model_str + ": HR@1 & HR@5 & HR@10 & MRR & NDCG@10")
print(np.mean(full_hr_1), np.mean(full_hr_5), np.mean(full_hr_10), np.mean(full_mrr), np.mean(full_ndcg))
print(np.std(full_hr_1), np.std(full_hr_5), np.std(full_hr_10), np.std(full_mrr), np.std(full_ndcg))

print("Degree-related performance: NDCG@10 & MRR")
print(np.mean(full_head_ndcg), np.mean(full_head_mrr), np.mean(full_tail_ndcg), np.mean(full_tail_mrr))
print(np.std(full_head_ndcg), np.std(full_head_mrr), np.std(full_tail_ndcg), np.std(full_tail_mrr))

