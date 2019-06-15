import numpy as np
import torch
from random import sample, shuffle
from sklearn.decomposition import PCA

def get_training_data(params, img_vec, words_vec):
    # Смотрим, кто будет лидером?
    X_img_ind = np.array([[[np.random.choice(range(len(img_vec)))
      for card in range(params['num_cards'])] for sample in range(params['num_samples'])]
                 for i in range(params['num_players'])])
    X_img = img_vec[np.array(X_img_ind)]
    
    ind_words = list(range(len(words_vec)))
    shuffle(ind_words)
    X_txt = np.repeat(words_vec[np.newaxis, ind_words[:params['num_words']], :],
                      params['num_samples'], axis = 0)
    
    # Обозначаем, какую карту выбрал игрок, который лидирует
    X_img_ind_to_choose = np.array([[[np.random.choice(params['num_cards'])]
                                           for i in range(params['num_samples'])]
                                   for j in range(params['num_players'])])
    y_what_card_leader_choose = np.apply_along_axis(lambda x: np.concatenate([np.zeros(x),np.ones(1),
                                                               np.zeros(params['num_cards']-x-1)]), 1,
                                 X_img_ind_to_choose[0,:,:])
    if params['num_players']<3:
        X_img, X_img_ind = X_img[0,:,:,:], X_img_ind[0,:,:]
    return(torch.FloatTensor(X_img), 
           torch.LongTensor(np.array(y_what_card_leader_choose)), 
           torch.FloatTensor(X_txt), 
           torch.FloatTensor(X_img_ind)) 



def pca_transfrom(words_vec_train, img_vec_train,
                  words_vec_test, img_vec_test,
                  n_components =0):
    if n_components==1024:
        return (words_vec_train, img_vec_train, words_vec_test, img_vec_test)
    pca = PCA(n_components)
    pca.fit(np.concatenate([words_vec_train, img_vec_train]))
    return (pca.transform(words_vec_train), pca.transform(img_vec_train),
           pca.transform(words_vec_test), pca.transform(img_vec_test))


