import os
import sys
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

import deepadr
from deepadr.dataset import *
from deepadr.utilities import *
from deepadr.chemfeatures import *
from deepadr.model_gnn_ogb import GNN, DeepAdr_SiameseTrf, ExpressionNN, DeepSynergy
from deepadr.model_attn_siamese import GeneEmbAttention, GeneEmbProjAttention
from ogb.graphproppred import Evaluator

import json
import functools

fdtype = torch.float32

torch.set_printoptions(precision=6)


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def F_score(a,b):
    return (2*a*b)/(a+b)

def generate_tp_hp(tp, hp, hp_names):
    tphp=deepcopy(tp)
    for i,n in enumerate(hp_names):
        tphp[n] = hp[i]
    return tphp


def build_predictions_df(ids, true_class, pred_class, prob_scores):

    prob_scores_dict = {}
    for i in range (prob_scores.shape[-1]):
        prob_scores_dict[f'prob_score_class{i}'] = prob_scores[:, i]

    df_dict = {
        'id': ids,
        'true_class': true_class,
        'pred_class': pred_class
    }
    df_dict.update(prob_scores_dict)
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.set_index('id', inplace=True)
    return predictions_df

def run_test(queue, used_dataset, gpu_num, tp, exp_dir, partition):
    print("gpu_num", gpu_num)

def run_exp(queue, used_dataset, gpu_num, tp, exp_dir, partition): #
    
    num_classes = 2
    
    targetdata_dir_raw = os.path.abspath(exp_dir + "/../../raw")
    targetdata_dir_processed = os.path.abspath(exp_dir + "/../../processed")
    
    state_dict_dir = os.path.join(exp_dir, 'modelstates')
    
    device_gpu = get_device(True, index=gpu_num)
    print("gpu:", device_gpu)
    
    # Serialize data into file:
    json.dump( tp, open( exp_dir + "/hyperparameters.json", 'w' ) )
    
    tp['nonlin_func'] = nn.ReLU()
    
    expression_scaler = TorchStandardScaler()
    expression_scaler.fit(used_dataset.data.expression[partition['train']])
    
    train_dataset = Subset(used_dataset, partition['train'])
    val_dataset = Subset(used_dataset, partition['validation'])
    test_dataset = Subset(used_dataset, partition['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=tp["batch_size"], shuffle=True, follow_batch=['x_a', 'x_b'])
    valid_loader = DataLoader(val_dataset, batch_size=tp["batch_size"], shuffle=False, follow_batch=['x_a', 'x_b'])
    test_loader = DataLoader(test_dataset, batch_size=tp["batch_size"], shuffle=False, follow_batch=['x_a', 'x_b'])
    
    loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}

    gnn_model = GNN(gnn_type = tp["gnn_type"], 
                num_layer = tp["num_layer"], 
                emb_dim = tp["emb_dim"], 
                drop_ratio = 0.5, 
                JK = "multilayer", #last
                graph_pooling = tp["graph_pooling"],
                virtual_node = False,
                with_edge_attr=False).to(device=device_gpu, dtype=fdtype)


    expression_model = DeepSynergy(D_in=(2*tp["emb_dim"])+tp["expression_input_size"],
                                   H1=tp['exp_H1'], H2=tp['exp_H2'], drop=tp['p_dropout']).to(device=device_gpu, dtype=fdtype)

    gene_attn_model = GeneEmbAttention(input_dim=tp["expression_input_size"]).to(device=device_gpu, dtype=fdtype)

    models_param = list(gnn_model.parameters()) + list(expression_model.parameters()) + list(gene_attn_model.parameters())


    model_name = "ogb"
    models = [(gnn_model, f'{model_name}_GNN'),
              (expression_model, f'{model_name}_Expression'),

              (gene_attn_model, f'{model_name}_GeneAttn'),
             ]
    

    y_weights = compute_class_weights(used_dataset.data.y[partition['train']])
    class_weights = torch.tensor(y_weights).type(fdtype).to(device_gpu)

    # from IPython.display import Javascript
    # display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

    num_iter = len(train_loader)  # num_train_samples/batch_size
    c_step_size = int(np.ceil(5*num_iter))  # this should be 2-10 times num_iter

    base_lr = tp['base_lr']
    max_lr = tp['max_lr_mul']*base_lr  # 3-5 times base_lr
    optimizer = torch.optim.Adam(models_param, weight_decay=tp["l2_reg"], lr=base_lr)
    cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=c_step_size,
                                                    mode='triangular', cycle_momentum=False)

    loss_nlll = torch.nn.NLLLoss(weight=class_weights, reduction='mean')  # negative log likelihood loss
    loss_contrastive = ContrastiveLoss(0.5, reduction='mean')
    
    
    valid_curve_aupr = []
    test_curve_aupr = []
    train_curve_aupr = []
    
    valid_curve_auc = []
    test_curve_auc = []
    train_curve_auc = []
    
    best_fscore = 0
    best_epoch = 0
      

    for epoch in range(tp["num_epochs"]):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        
        for m, m_name in models:
            m.train()

        for i_batch, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            batch = batch.to(device_gpu)

            h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
            h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)
            
            expression_norm = expression_scaler.transform_ondevice(batch.expression, device=device_gpu) 
            h_e, _ = gene_attn_model(expression_norm.type(fdtype))
            
            triplet = torch.cat([h_a, h_b, h_e], axis=-1)

            logsoftmax_scores = expression_model(triplet)

            loss = loss_nlll(logsoftmax_scores, batch.y.type(torch.long))            

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            cyc_scheduler.step() # after each batch step the scheduler
            optimizer.zero_grad()  # Clear gradients.

        print('Evaluating...')

        perfs = {}

        for dsettype in ["train", "valid"]:
            for m, m_name in models:
                m.eval()

            pred_class = []
            ref_class = []
            prob_scores = []
            
            l_ids = []
           


        #     for data in loader:  # Iterate in batches over the training/test dataset.
            for i_batch, batch in enumerate(tqdm(loaders[dsettype], desc="Iteration")):
                batch = batch.to(device_gpu)
                h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
                h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)

                expression_norm = expression_scaler.transform_ondevice(batch.expression, device=device_gpu) 
                h_e, _ = gene_attn_model(expression_norm.type(fdtype))


                triplet = torch.cat([h_a, h_b, h_e], axis=-1)
                
                logsoftmax_scores = expression_model(triplet)


                __, y_pred_clss = torch.max(logsoftmax_scores, -1)

                y_pred_prob  = torch.exp(logsoftmax_scores.detach().cpu()).numpy()

                pred_class.extend(y_pred_clss.view(-1).tolist())
                ref_class.extend(batch.y.view(-1).tolist())
                prob_scores.append(y_pred_prob)
                l_ids.extend(batch.id.view(-1).tolist())

            prob_scores_arr = np.concatenate(prob_scores, axis=0)

            dset_perf = perfmetric_report(pred_class, ref_class, prob_scores_arr[:,1], epoch,
                                          outlog = os.path.join(exp_dir, dsettype + ".log"))
            
            perfs[dsettype] = dset_perf
            
            if (dsettype=="valid"):
                
                fscore = F_score(perfs['valid'].s_aupr, perfs['valid'].s_auc)
                if (fscore > best_fscore):
                    best_fscore = fscore
                    best_epoch = epoch
                    
                    for m, m_name in models:
                        torch.save(m.state_dict(), os.path.join(state_dict_dir, '{}.pkl'.format(m_name)))

        print({'Train': perfs['train'], 'Validation': perfs['valid']})

        
        train_curve_aupr.append(perfs['train'].s_aupr)
        valid_curve_aupr.append(perfs['valid'].s_aupr)
        test_curve_aupr.append(0.0)

        
        train_curve_auc.append(perfs['train'].s_auc)
        valid_curve_auc.append(perfs['valid'].s_auc)
        test_curve_auc.append(0.0)


    print('Finished training and validating!')
        
        
    for dsettype in ["test"]:
        
        if(len(os.listdir(state_dict_dir)) > 0):  # load state dictionary of saved models
            for m, m_name in models:
                m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device_gpu))

        
        for m, m_name in models:
            m.eval()

        pred_class = []
        ref_class = []
        prob_scores = []

        l_ids = []


    #     for data in loader:  # Iterate in batches over the training/test dataset.
        for i_batch, batch in enumerate(tqdm(loaders[dsettype], desc="Iteration")):
            batch = batch.to(device_gpu)
            h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
            h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)

            expression_norm = expression_scaler.transform_ondevice(batch.expression, device=device_gpu) 
            h_e, _ = gene_attn_model(expression_norm.type(fdtype))


            triplet = torch.cat([h_a, h_b, h_e], axis=-1)

            logsoftmax_scores = expression_model(triplet)


            __, y_pred_clss = torch.max(logsoftmax_scores, -1)

            y_pred_prob  = torch.exp(logsoftmax_scores.detach().cpu()).numpy()

            pred_class.extend(y_pred_clss.view(-1).tolist())
            ref_class.extend(batch.y.view(-1).tolist())
            prob_scores.append(y_pred_prob)
            l_ids.extend(batch.id.view(-1).tolist())

        prob_scores_arr = np.concatenate(prob_scores, axis=0)

        dset_perf = perfmetric_report(pred_class, ref_class, prob_scores_arr[:,1], epoch,
                                      outlog = os.path.join(exp_dir, dsettype + ".log"))

        perfs[dsettype] = dset_perf

        if (dsettype=="test"):

            predictions_df = build_predictions_df(l_ids, ref_class, pred_class, prob_scores_arr)
            predictions_df.to_csv(os.path.join(exp_dir, 'predictions', f'epoch_{epoch}_predictions_{dsettype}.csv'))
            
        print({'Test': perfs['test']})

        test_curve_aupr.pop()
        test_curve_aupr.append(perfs['test'].s_aupr)

        test_curve_auc.pop()
        test_curve_auc.append(perfs['test'].s_auc)

    print('Finished testing!')

    df_curves = pd.DataFrame(np.array([train_curve_aupr, valid_curve_aupr, test_curve_aupr,
                                       train_curve_auc, valid_curve_auc, test_curve_auc]).T)
    df_curves.columns = ['train_aupr', 'valid_aupr', 'test_aupr', 'train_auc', 'valid_auc', 'test_auc']
    df_curves.index.name = "epoch"
    df_curves.to_csv(exp_dir + "/curves.csv")
    sns.lineplot(data=df_curves).figure.savefig(exp_dir + "/curves.png")
    
    queue.put(gpu_num)