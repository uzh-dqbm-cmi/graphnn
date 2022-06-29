import os
from os import path

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
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch.utils.data import DataLoader

# os.chdir('..')
import deepadr
from deepadr.dataset import *
from deepadr.utilities import *
from deepadr.run_workflow import *
from deepadr.chemfeatures import *
# from deepadr.hyphelper import *
# from deepadr.model_gnn import GCN as testGCN
from deepadr.model_gnn_ogb import GNN, DeepAdr_SiameseTrf, ExpressionNN, DeepSynergy
# from deepadr.model_attn_siamese import *
from ogb.graphproppred import Evaluator
# os.chdir(cwd)

import json
import functools

# imports from captum library
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    DeepLiftShap,
    GradientShap,
    NoiseTunnel,
    FeatureAblation,
    Saliency,
    InputXGradient,
    Deconvolution,
    FeaturePermutation
)

fdtype = torch.float32


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


def run_exp_flat(queue, used_dataset, gpu_num, tp, exp_dir, partition): #
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=tp["batch_size"], shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=tp["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=tp["batch_size"], shuffle=False)
    
    loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}

    deepsynergy_model = ExpressionNN(D_in=tp['deepsynergy_input_size']).to(device=device_gpu, dtype=fdtype)
    
    print("DS model:\n", deepsynergy_model)

    models_param = list(deepsynergy_model.parameters())


    model_name = "deepsynergy"
    models = [(deepsynergy_model, f'{model_name}_model')]
    #models

    y_weights = ReaderWriter.read_data(os.path.join(targetdata_dir_raw, 'y_weights.pkl'))
    class_weights = torch.tensor(y_weights).to(device=device_gpu, dtype=fdtype)
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

        for i_batch, (batch_x, batch_y, batch_ids) in enumerate(tqdm(train_loader, desc="Iteration")):
            
            batch_x = batch_x.to(device=device_gpu, dtype=fdtype)
            batch_y = batch_y.to(device=device_gpu, dtype=fdtype)


#             h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
#             h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)

#             z_e, _ = transformer_model(torch.unsqueeze(batch.expression.type(fdtype), dim=1))

            logsoftmax_scores = deepsynergy_model(batch_x)

#             logsoftmax_scores, dist = siamese_model(h_a, h_b, z_e)
    #         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
    #         loss = criterion(out, samples_batch.y)  # Compute the loss.
    #         print(pd.Series(batch.y.cpu()).value_counts())
            loss = loss_nlll(logsoftmax_scores, batch_y.type(torch.long))            
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
            
            fattn_w_scores_e_ids = []
            l_ids = []
           


        #     for data in loader:  # Iterate in batches over the training/test dataset.
#             for i_batch, batch in enumerate(tqdm(loaders[dsettype], desc="Iteration")):
            for i_batch, (batch_x, batch_y, batch_ids) in enumerate(tqdm(loaders[dsettype], desc="Iteration")):
                
                batch_x = batch_x.to(device=device_gpu, dtype=fdtype)
                batch_y = batch_y.to(device=device_gpu, dtype=fdtype)

                logsoftmax_scores = deepsynergy_model(batch_x)
          

                __, y_pred_clss = torch.max(logsoftmax_scores, -1)

                y_pred_prob  = torch.exp(logsoftmax_scores.detach().cpu()).numpy()

                pred_class.extend(y_pred_clss.view(-1).tolist())
                ref_class.extend(batch_y.view(-1).tolist())
                prob_scores.append(y_pred_prob)
                l_ids.extend(batch_ids.view(-1).tolist())

            prob_scores_arr = np.concatenate(prob_scores, axis=0)

            dset_perf = perfmetric_report(pred_class, ref_class, prob_scores_arr[:,1], epoch,
                                          outlog = os.path.join(exp_dir, dsettype + ".log"))
            
            perfs[dsettype] = dset_perf
            
            if (dsettype=="test"):
                
                fscore = F_score(perfs['test'].s_aupr, perfs['test'].s_auc)
                if (fscore > best_fscore):
                    best_fscore = fscore
                    best_epoch = epoch
                
                   
                
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
    
    for m, m_name in models:
        torch.save(m.state_dict(), os.path.join(exp_dir, 'modelstates', f'{m_name}_statedict.pt'))

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
    
def run_attribution(queue, x_np_norm, gpu_num, tp, exp_dir, partition, labels, attrAlgName): #
    
    num_classes = 2
    
    targetdata_dir_raw = os.path.abspath(exp_dir + "/../../raw")
    targetdata_dir_processed = os.path.abspath(exp_dir + "/../../processed")
    
    device_gpu = get_device(True, index=gpu_num)
    print("gpu:", device_gpu)
    
    # Serialize data into file:
    json.dump( tp, open( exp_dir + "/hyperparameters.json", 'w' ) )
    
    tp['nonlin_func'] = nn.ReLU()
    
    deepsynergy_model = ExpressionNN(D_in=tp['deepsynergy_input_size']).to(device=device_gpu, dtype=fdtype)
    
    print("DS model:\n", deepsynergy_model)

    models_param = list(deepsynergy_model.parameters())


    model_name = "deepsynergy"
    models = [(deepsynergy_model, f'{model_name}_model')]
    #models 
    
    for m, m_name in models:
        model_path = os.path.join(exp_dir, 'modelstates', f'{m_name}_statedict.pt')
        
        if path.isfile(model_path):
            print('Loading pre-trained model from: {}'.format(model_path))
            m.load_state_dict(torch.load(model_path)) #, map_location=device
        else:   
            print('Missing model states, please train models first.')
            return
    
#     train_dataset = Subset(used_dataset, partition['train'])
#     val_dataset = Subset(used_dataset, partition['validation'])
#     test_dataset = Subset(used_dataset, partition['test'])
    
#     train_loader = DataLoader(train_dataset, batch_size=tp["batch_size"], shuffle=True)
#     valid_loader = DataLoader(val_dataset, batch_size=tp["batch_size"], shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=tp["batch_size"], shuffle=False)
    
#     test_partition_TP_zeros, test_partition_TP_ones = partition
    test_partition_TP = partition
    
    test_correct_labels = np.take(labels, test_partition_TP)
    test_correct_labels_tensor = torch.from_numpy(test_correct_labels).to(device=device_gpu, dtype=torch.int64)
    
    for labels in ['all']:
        if (labels == 'zeros'):
            test_partition_TP = test_partition_TP_zeros
            target = 0
        elif (labels == 'ones'):
            test_partition_TP = test_partition_TP_ones
            target = 1
        else:
            target = test_correct_labels_tensor
    
#     test_correct_labels_zeros = np.take(labels, test_partition_TP_zeros)
#     test_correct_labels_zeros_tensor = torch.from_numpy(test_correct_labels_zeros).to(device=device_gpu, dtype=torch.int64)
    
#     test_correct_labels_ones = np.take(labels, test_partition_TP_ones)
#     test_correct_labels_ones_tensor = torch.from_numpy(test_correct_labels_ones).to(device=device_gpu, dtype=torch.int64)
    
        test_features = np.take(x_np_norm, test_partition_TP, axis=0)
        test_input_tensor = torch.from_numpy(test_features).to(device=device_gpu, dtype=fdtype)
        n_test_samples = test_input_tensor.size()[0]

        test_min, _ = torch.min(test_input_tensor, dim=0)
        test_max, _ = torch.max(test_input_tensor, dim=0)

        epsilon = 1e-3

        test_min_bline = (test_min-epsilon).repeat(n_test_samples, 1)
        test_max_bline = (test_max+epsilon).repeat(n_test_samples, 1)

    #     print("Testbline shape:", test_bline.size())
    #     print("test_min_bline range:", torch.min(test_min_bline), torch.max(test_min_bline))


    #     loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}



        print("Starting attr calc...")

#         attrAlgName = 'DeepLiftShap'

        if (attrAlgName == 'IntegratedGradients'):
            attrAlg = IntegratedGradients(models[0][0])
        elif (attrAlgName == 'DeepLiftShap'):
            attrAlg = DeepLiftShap(models[0][0])
        elif (attrAlgName == 'DeepLift'):
            attrAlg = DeepLift(models[0][0])
        elif (attrAlgName == 'GradientShap'):
            attrAlg = GradientShap(models[0][0])
    #     DeepLift,
    #     DeepLiftShap,
    #     GradientShap,
    #     NoiseTunnel,
    #     FeatureAblation,
    #     Saliency,
    #     InputXGradient,
    #     Deconvolution,
    #     FeaturePermutation

        for bline in ['min']:
            if (bline == 'min'):
                test_bline = test_min_bline
            else:
                test_bline = test_max_bline

            if (attrAlgName == "IntegratedGradients"):
                attributions, deltas = attrAlg.attribute(inputs=test_input_tensor,
                                                        baselines=test_bline,
    #                                                     target=test_correct_labels_tensor,
                                                        target=target,
                                                        return_convergence_delta=True,
                                                        internal_batch_size=1,
                                                        n_steps=200)
#             elif (attrAlgName == "GradientShap"):
#                 attributions, deltas = attrAlg.attribute(inputs=test_input_tensor,
#                                                         baselines=test_bline,
#     #                                                     target=test_correct_labels_tensor,
#                                                         target=target,
#                                                         return_convergence_delta=True)
            else:
                attributions, deltas = attrAlg.attribute(inputs=test_input_tensor,
                                                        baselines=test_bline,
    #                                                     target=test_correct_labels_tensor,
                                                        target=target,
                                                        return_convergence_delta=True)

            ReaderWriter.dump_tensor(attributions, os.path.join(exp_dir, 'attributions', f'{attrAlgName}_attributions_{bline}_{labels}.tensor'))

            ReaderWriter.dump_tensor(deltas, os.path.join(exp_dir, 'attributions', f'{attrAlgName}_deltas_{bline}_{labels}.tensor'))

        #     attributions = attributions.detach().cpu().numpy()
        #     print("attr shape:", attributions.shape)

        #     q_attr.put(attributions)
    
    queue.put(gpu_num)