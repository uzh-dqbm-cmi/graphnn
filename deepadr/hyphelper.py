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
from deepadr.model_gnn_ogb import GNN, DeepAdr_SiameseTrf, ExpressionNN
# from deepadr.model_attn_siamese import *
from ogb.graphproppred import Evaluator
# os.chdir(cwd)

import json

fdtype = torch.float32

def generate_tp_hp(tp, hp, hp_names):
    tphp=deepcopy(tp)
    for i,n in enumerate(hp_names):
#         print(n)
#         print(hp[i])
        tphp[n] = hp[i]
    return tphp


def run_exp(queue, used_dataset, gpu_num, tp, exp_dir, partition): #
    
    targetdata_dir_raw = os.path.abspath(exp_dir + "/../../raw")
    
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
                    JK = "multilayer", #last
                    graph_pooling = tp["graph_pooling"],
                    virtual_node = False,
                    with_edge_attr=False).to(device=device_gpu, dtype=fdtype)

    transformer_model = DeepAdr_Transformer(input_size=tp["expression_input_size"],
                                            input_embed_dim=tp["input_embed_dim"],
                                            num_attn_heads=tp["num_attn_heads"],
                                            mlp_embed_factor=tp["mlp_embed_factor"],
                                            nonlin_func=tp["nonlin_func"],
                                            pdropout=tp["p_dropout"],
                                            num_transformer_units=tp["num_transformer_units"],
                                            pooling_mode=tp["pooling_mode"]).to(device=device_gpu, dtype=fdtype)

    # expression_model = ExpressionNN(D_in=tp["expression_input_size"],
    #                                 H1=tp["exp_H1"], H2=tp["exp_H2"],
    #                                 D_out=tp["expression_dim"], drop=0.5).to(device=device_gpu, dtype=fdtype)

    siamese_model = DeepAdr_SiameseTrf(input_dim=tp["emb_dim"],
                                       dist=tp["dist_opt"],
                                       expression_dim=tp["expression_input_size"],
                                       num_classes=2).to(device=device_gpu, dtype=fdtype)

    # models_param = list(gnn_model.parameters()) + list(transformer_model.parameters()) + list(siamese_model.parameters()) + list(expression_model.parameters())
    models_param = list(gnn_model.parameters()) + list(transformer_model.parameters()) + list(siamese_model.parameters())


    model_name = "ogb"
    models = [(gnn_model, f'{model_name}_GNN'),
              (transformer_model, f'{model_name}_Transformer'),
              (siamese_model, f'{model_name}_Siamese'),
    #           (expression_model, f'{model_name}_Expression'),
    #           (lassonet_model, f'{model_name}_LassoNet')
             ]
    #models

    y_weights = ReaderWriter.read_data(os.path.join(targetdata_dir_raw, 'y_weights.pkl'))
    class_weights = torch.tensor(y_weights).type(fdtype).to(device_gpu)
    class_weights

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
    
    
    valid_curve = []
    test_curve = []
    train_curve = []

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

            transformer_input = torch.unsqueeze(batch.expression.type(fdtype), dim=1)
            z_e, fattn_w_scores_e = transformer_model(transformer_input)

            logsoftmax_scores, dist = siamese_model(h_a, h_b, z_e)
    #         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
    #         loss = criterion(out, samples_batch.y)  # Compute the loss.
    #         print(pd.Series(batch.y.cpu()).value_counts())
            cl = loss_nlll(logsoftmax_scores, batch.y.type(torch.long))            
            dl = loss_contrastive(dist.reshape(-1), batch.y.type(fdtype))          
            loss = tp["loss_w"]*cl + (1-tp["loss_w"])*dl
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

        #     for data in loader:  # Iterate in batches over the training/test dataset.
            for i_batch, batch in enumerate(tqdm(loaders[dsettype], desc="Iteration")):
                batch = batch.to(device_gpu)
                h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
                h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)

                transformer_input = torch.unsqueeze(batch.expression.type(fdtype), dim=1)
                z_e, fattn_w_scores_e = transformer_model(transformer_input)

                logsoftmax_scores, dist = siamese_model(h_a, h_b, z_e)

                __, y_pred_clss = torch.max(logsoftmax_scores, -1)

                y_pred_prob  = torch.exp(logsoftmax_scores.detach().cpu()).numpy()

                pred_class.extend(y_pred_clss.view(-1).tolist())
                ref_class.extend(batch.y.view(-1).tolist())
                prob_scores.append(y_pred_prob)

            prob_scores_arr = np.concatenate(prob_scores, axis=0)

            dset_perf = perfmetric_report(pred_class, ref_class, prob_scores_arr[:,1], epoch,
                                          outlog = os.path.join(exp_dir, dsettype + ".log"))
            
            perfs[dsettype] = dset_perf
      

        print({'Train': perfs['train'], 'Validation': perfs['valid'], 'Test': perfs['test']})

        train_curve.append(perfs['train'].s_aupr)
        valid_curve.append(perfs['valid'].s_aupr)
        test_curve.append(perfs['test'].s_aupr)

    # if 'classification' in dataset.task_type:
    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    # else:
    #     best_val_epoch = np.argmin(np.array(valid_curve))
    #     best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    df_curves = pd.DataFrame(np.array([train_curve, valid_curve, test_curve]).T)
    df_curves.columns = ['train', 'valid', 'test']
    df_curves.index.name = "epoch"
    df_curves.to_csv(exp_dir + "/curves.csv")
    sns.lineplot(data=df_curves).figure.savefig(exp_dir + "/curves.png")
    
    queue.put(gpu_num)