import math
import torch
import numpy as np

from tqdm import tqdm
from time import sleep
from lib.model_classes import ImaginariumModel
from lib.data_preprocessing import get_training_data, pca_transfrom
from lib.gumbel_softmax import gumbel_softmax


def learning_pipeline(img_vec, words_vec, params, config_train):
    params_inference = params
    params_inference['inference'] = True
        
    model = ImaginariumModel(params)
    #model_inference = ImaginariumModel(params_inference)
    
    loss_fn = torch.nn.NLLLoss()
    accuracy_list, loss_list = [], []
    
    words_vec_train = words_vec
    words_vec_test = words_vec
    img_vec_train = img_vec[:int(img_vec.shape[0]*0.8),:]
    img_vec_test = img_vec[int(img_vec.shape[0]*0.8):,:]
    
    words_vec_train_pca, img_vec_train_pca, words_vec_test_pca, img_vec_test_pca  = pca_transfrom(
    words_vec_train, img_vec_train, words_vec_test, img_vec_test, n_components = params['len_emb'])

    
    with tqdm(total=10, bar_format="{postfix[0]} {postfix[1][proc]} {postfix[2][accuracy]}",
              postfix=["Procent done", dict(proc=0), dict(accuracy=0)]) as tqdm_log:

        # Оптимизируем сетку
        for num_epoch in range(config_train['num_batches']):
            # генерируем метрики
            X_img, y_what_card_leader_choose, X_txt, X_img_ind = get_training_data(
                params, img_vec_train_pca, words_vec_train_pca)
            X_img_test, y_what_card_leader_choose_test, X_txt_test, X_img_ind_test = get_training_data(
                params, img_vec_test_pca, words_vec_test_pca)
            # считаем метрики
            y_pred_test = model(X_img_test, X_txt_test,
                                y_what_card_leader_choose_test)

            y_pred_ind_test = np.argmax(y_pred_test.detach().numpy(),axis = 1)
            target_test = np.argmax(y_what_card_leader_choose_test, axis = 1)

            
            accuracy = np.mean([x == y for x,y in zip(y_pred_ind_test, target_test)])
            accuracy_list.append(accuracy)

            # Оптимизируем
            optimizer = torch.optim.SGD(model.parameters(), lr=1)
            clr = cyclical_lr(config_train['step_size'],
                              min_lr = config_train['end_lr'] / config_train['factor'],
                              max_lr = config_train['end_lr'])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

            # текут градиенты
            y_pred = model(X_img, X_txt, y_what_card_leader_choose)

            y_pred_ind = np.argmax(y_pred.detach().numpy(),axis = 1)
            target = np.argmax(y_what_card_leader_choose, axis = 1)
            loss = loss_fn(y_pred, target)
            loss_list.append(loss.item())
            
            if min(loss_list) == loss:
                torch.save(model.state_dict(), 'models/{0}.pth'.format(config_train['model_name']))

            optimizer.zero_grad()
            loss.backward()
            #scheduler.step()
            optimizer.step()

            tqdm_log.postfix[1]["proc"] = '{0}%'.format(str(round(num_epoch*1.0 / config_train['num_batches'], 2)))
            tqdm_log.postfix[2]["accuracy"] = '         accuracy = {0}'.format(str(round(accuracy, 2)))
            #tqdm_log.postfix[2]["time"] = '         time_remain = {0}'.format(str(time_remain))
            tqdm_log.update()

    result = {
        'model_name': config_train['model_name'],
        'accuracy_list': accuracy_list,
        'loss_list': loss_list,
        'model': model,
        'base_accuracy': accuracy_list[0],
        'config_train': config_train,
        'config_network': params,
    }
    return result


def cyclical_lr(stepsize, min_lr=3e-2, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda