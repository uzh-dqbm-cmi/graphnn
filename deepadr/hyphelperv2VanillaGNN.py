import os
import sys
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
# import ogb
from tqdm import tqdm
# import hiplot as hip
from copy import deepcopy

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

# os.chdir('..')
import deepadr
from deepadr.dataset import *
from deepadr.utilities import *
from deepadr.run_workflow import *
from deepadr.chemfeatures import *
# from deepadr.hyphelper import *
# from deepadr.model_gnn import GCN as testGCN
from deepadr.model_gnn_ogb import GNN, DeepAdr_SiameseTrf, ExpressionNN, DeepSynergy
from deepadr.model_attn_siamese import GeneEmbAttention, GeneEmbProjAttention
from ogb.graphproppred import Evaluator
# os.chdir(cwd)

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
#         print(n)
#         print(hp[i])
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


def run_exp_vanilla(queue, used_dataset, gpu_num, tp, exp_dir, partition): #
    
    num_classes = 2
    
    targetdata_dir_raw = os.path.abspath(exp_dir + "/../../raw")
    targetdata_dir_processed = os.path.abspath(exp_dir + "/../../processed")
    
    device_gpu = get_device(True, index=gpu_num)
    print("gpu:", device_gpu)
    
    # Serialize data into file:
    json.dump( tp, open( exp_dir + "/hyperparameters.json", 'w' ) )
    
    tp['nonlin_func'] = nn.ReLU()
    
    train_dataset = Subset(used_dataset, partition['train'])
    val_dataset = Subset(used_dataset, partition['validation'])
    test_dataset = Subset(used_dataset, partition['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=tp["batch_size"], shuffle=True, follow_batch=['x_a', 'x_b'])
    valid_loader = DataLoader(val_dataset, batch_size=tp["batch_size"], shuffle=False, follow_batch=['x_a', 'x_b'])
    test_loader = DataLoader(test_dataset, batch_size=tp["batch_size"], shuffle=False, follow_batch=['x_a', 'x_b'])
    
    loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}

    gnn_model = GNN(gnn_type = tp["gnn_type"], 
#                 num_tasks = dataset.num_classes, 
                num_layer = tp["num_layer"], 
                emb_dim = tp["emb_dim"], 
                drop_ratio = 0.5, 
                JK = "last",
                graph_pooling = tp["graph_pooling"],
                virtual_node = False,
                with_edge_attr=False).to(device=device_gpu, dtype=fdtype)

#     transformer_model = DeepAdr_Transformer(input_size=tp["expression_input_size"],
#                                         input_embed_dim=tp["input_embed_dim"],
#                                         num_attn_heads=tp["num_attn_heads"],
#                                         mlp_embed_factor=tp["mlp_embed_factor"],
#                                         nonlin_func=tp["nonlin_func"],
#                                         pdropout=tp["p_dropout"],
#                                         num_transformer_units=tp["num_transformer_units"],
#                                         pooling_mode=tp["pooling_mode"],
#                                         gene_embed_dim=tp['gene_embed_dim']).to(device=device_gpu, dtype=fdtype)

    expression_model = DeepSynergy(D_in=(2*tp["emb_dim"])+tp["expression_input_size"],
                                   H1=tp['exp_H1'], H2=tp['exp_H2'], drop=tp['p_dropout']).to(device=device_gpu, dtype=fdtype)

#     expression_model = DeepSynergy(D_in=(2*tp["emb_dim"])+tp["gene_embed_dim"],
#                                H1=tp['exp_H1'], H2=tp['exp_H2'], drop=tp['p_dropout']).to(device=device_gpu, dtype=fdtype)

#     siamese_model = DeepAdr_SiameseTrf(input_dim=tp["emb_dim"],
#                                    dist=tp["dist_opt"],
#                                    expression_dim=tp["emb_dim"],
#                                    gene_embed_dim=tp['gene_embed_dim'],
#                                    num_classes=num_classes).to(device=device_gpu, dtype=fdtype)

#     gene_attn_model = GeneEmbProjAttention(input_dim=tp["expression_input_size"],
#                                            nonlin_func=tp['nonlin_func'],
#                                            gene_embed_dim=tp['gene_embed_dim']).to(device=device_gpu, dtype=fdtype)

    # models_param = list(gnn_model.parameters()) + list(transformer_model.parameters()) + list(siamese_model.parameters()) + list(expression_model.parameters())
    models_param = list(gnn_model.parameters()) + list(expression_model.parameters())


    model_name = "ogb"
    models = [(gnn_model, f'{model_name}_GNN'),
#               (transformer_model, f'{model_name}_Transformer'),
              (expression_model, f'{model_name}_Expression'),
#               (siamese_model, f'{model_name}_Siamese'),
    #           (lassonet_model, f'{model_name}_LassoNet')
#               (gene_attn_model, f'{model_name}_GeneAttn'),
             ]
    #models

    y_weights = ReaderWriter.read_data(os.path.join(targetdata_dir_raw, 'y_weights.pkl'))
    class_weights = torch.tensor(y_weights).type(fdtype).to(device_gpu)
#     class_weights

    # from IPython.display import Javascript
    # display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

    num_iter = len(train_loader)  # num_train_samples/batch_size
    c_step_size = int(np.ceil(5*num_iter))  # this should be 2-10 times num_iter

    base_lr = tp['base_lr']
    max_lr = tp['max_lr_mul']*base_lr  # 3-5 times base_lr
    optimizer = torch.optim.Adam(models_param, weight_decay=tp["l2_reg"], lr=base_lr)
    cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=c_step_size,
                                                    mode='triangular', cycle_momentum=False)
    # optimizer = torch.optim.Adam(models_param, lr=0.001)
    # criterion = torch.nn.CrossEntropyLoss()

    # loss_nlll = torch.nn.NLLLoss(weight=class_weights, reduction='mean')  # negative log likelihood loss
    loss_nlll = torch.nn.NLLLoss(weight=class_weights, reduction='mean')  # negative log likelihood loss
    loss_contrastive = ContrastiveLoss(0.5, reduction='mean')
    # loss_mse = torch.nn.MSELoss()  # this is for regression mean squared loss


#     # evaluator = Evaluator(DSdataset_name)
    
    
    valid_curve_aupr = []
    test_curve_aupr = []
    train_curve_aupr = []
    
    valid_curve_auc = []
    test_curve_auc = []
    train_curve_auc = []
    
    best_fscore = 0
    best_epoch = 0
      

    for epoch in range(tp["num_epochs"]):
    # for epoch in range(60,70):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        
        for m, m_name in models:
            m.train()

        for i_batch, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            batch = batch.to(device_gpu)

            h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
            h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)
            
#             h_e, _ = gene_attn_model(batch.expression.type(fdtype))
#             h_e, _ = gene_attn_model(batch.expression.type(fdtype), h_a, h_b)
            h_e = batch.expression.type(fdtype)


            
            triplet = torch.cat([h_a, h_b, h_e], axis=-1)

#             z_e = expression_model(torch.unsqueeze(batch.expression.type(fdtype), dim=1))
            logsoftmax_scores = expression_model(triplet)


#             logsoftmax_scores, dist = siamese_model(h_a, h_b, z_e)
    #         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
    #         loss = criterion(out, samples_batch.y)  # Compute the loss.
    #         print(pd.Series(batch.y.cpu()).value_counts())
            loss = loss_nlll(logsoftmax_scores, batch.y.type(torch.long))            
#             dl = loss_contrastive(dist.reshape(-1), batch.y.type(fdtype))          
#             loss = tp["loss_w"]*cl + (1-tp["loss_w"])*dl
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            cyc_scheduler.step() # after each batch step the scheduler
            optimizer.zero_grad()  # Clear gradients.

        print('Evaluating...')

        perfs = {}

        for dsettype in ["train", "test", "valid"]:
            for m, m_name in models:
                m.eval()

            pred_class = []
            ref_class = []
            prob_scores = []
            
#             fattn_w_scores_e_ids = []
            l_ids = []
           


        #     for data in loader:  # Iterate in batches over the training/test dataset.
            for i_batch, batch in enumerate(tqdm(loaders[dsettype], desc="Iteration")):
                batch = batch.to(device_gpu)
                
                h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
                h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)

    #             h_e, _ = gene_attn_model(batch.expression.type(fdtype))
    #             h_e, _ = gene_attn_model(batch.expression.type(fdtype), h_a, h_b)
                h_e = batch.expression.type(fdtype)



                triplet = torch.cat([h_a, h_b, h_e], axis=-1)

    #             z_e = expression_model(torch.unsqueeze(batch.expression.type(fdtype), dim=1))
                logsoftmax_scores = expression_model(triplet)

                
#                 ids = batch.id#.unsqueeze(1)
                
#                 print("ids:", ids.shape)
                
#                 print("np ids:", ids.detach().cpu().numpy().shape)
                
#                 if (dsettype=="test"):
#                     fattn_w_scores_e_ids.append(torch.cat((batch.id.unsqueeze(1), fattn_w_scores_e), 1))


#                 logsoftmax_scores, dist = siamese_model(h_a, h_b, z_e)

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
                
                fscore = F_score(perfs['test'].s_aupr, perfs['test'].s_auc)
                if (fscore > best_fscore):
                    best_fscore = fscore
                    best_epoch = epoch
                
#                     fattn_w_scores_e_ids_np = torch.cat(fattn_w_scores_e_ids).detach().cpu().numpy()
#                     df_fattn_w_scores_e_ids = pd.DataFrame(fattn_w_scores_e_ids_np)
#                     df_fattn_w_scores_e_ids.columns = ["id"] + ["gex"+str(i) for i in range(int(tp['expression_input_size']))]
#                     df_fattn_w_scores_e_ids.to_csv(os.path.join(exp_dir, "fattn_w_scores_e_ids_test" + ".csv"))
                
#                 np_ids = ids.detach().cpu().numpy()
                
#                 print("np_ids.shape: ", len(np_ids))
#                 print("ref_class.shape: ", len(ref_class))
#                 print("pred_class.shape: ", len(pred_class))
#                 print("prob_scores_arr.shape: ", len(prob_scores_arr))
                
                predictions_df = build_predictions_df(l_ids, ref_class, pred_class, prob_scores_arr)
                predictions_df.to_csv(os.path.join(exp_dir, 'predictions', f'epoch_{epoch}_predictions_{dsettype}.csv'))

      

        print({'Train': perfs['train'], 'Validation': perfs['valid'], 'Test': perfs['test']})

        train_curve_aupr.append(perfs['train'].s_aupr)
        valid_curve_aupr.append(perfs['valid'].s_aupr)
        test_curve_aupr.append(perfs['test'].s_aupr)
        
        train_curve_auc.append(perfs['train'].s_auc)
        valid_curve_auc.append(perfs['valid'].s_auc)
        test_curve_auc.append(perfs['test'].s_auc)
       

    # if 'classification' in dataset.task_type:
#     best_val_epoch = np.argmax(np.array(valid_curve_aupr))
#     best_train = max(train_curve_aupr)
    # else:
    #     best_val_epoch = np.argmin(np.array(valid_curve))
    #     best_train = min(train_curve)

    print('Finished training!')
#     print('Best validation score: {}'.format(train_curve_aupr[best_val_epoch]))
#     print('Test score: {}'.format(test_curve_aupr[best_val_epoch]))

    df_curves = pd.DataFrame(np.array([train_curve_aupr, valid_curve_aupr, test_curve_aupr,
                                       train_curve_auc, valid_curve_auc, test_curve_auc]).T)
    df_curves.columns = ['train_aupr', 'valid_aupr', 'test_aupr', 'train_auc', 'valid_auc', 'test_auc']
    df_curves.index.name = "epoch"
    df_curves.to_csv(exp_dir + "/curves.csv")
    sns.lineplot(data=df_curves).figure.savefig(exp_dir + "/curves.png")
    
    queue.put(gpu_num)