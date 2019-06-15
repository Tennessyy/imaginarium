import torch

class ImaginariumModel(torch.nn.Module):
    def __init__(self, params):
        super(ImaginariumModel, self).__init__()
        self._params = params
        self._linear_words1 = torch.nn.Linear(params['len_emb'], params['len_dense'])
        self._linear_words2 = torch.nn.Linear(params['len_dense'], params['len_dense'])
        self._linear_img1 = torch.nn.Linear(params['len_emb'], params['len_dense'])
        self._linear_img2 = torch.nn.Linear(params['len_emb'], params['len_dense'])
        self._distance2axis_words = torch.nn.modules.distance.CosineSimilarity(dim = 2)
        self._distance2axis_final = torch.nn.modules.distance.CosineSimilarity(dim = 2)
        self._softmax_final = torch.nn.Softmax()
        self._softmax_words = torch.nn.Softmax()
        self._relu = torch.nn.ReLU()
        self._softmax1dim = torch.nn.Softmax(dim=1)
        self._softmax2dim = torch.nn.Softmax(dim=2)
        self._combination_words = torch.nn.Sequential(self._linear_words1, self._softmax2dim, self._linear_words2)
        self._combination_img = torch.nn.Sequential(self._linear_img1, self._softmax2dim, self._linear_img2)

    def forward(self, x_img_dense, x_txt_dense, y_what_card_leader_choose):
       
        #if not self._params['inference']:
        if self._params['img_layer']:
            if self._params['comb_layer']:
                x_img_dense = self._combination_img(x_img_dense)
            else:
                x_img_dense = self._linear_img1(x_img_dense)

        if self._params['words_layer']:
            if same_imf_words_layer:
                if seld._params['comb_layer']:
                    x_txt_dense = self._combination_img(x_txt_dense)
                else:
                    x_txt_dense = self._linear_img1(x_txt_dense)
            else:
                if self._params['comb_layer']:
                    x_txt_dense = self._combination_words(x_txt_dense)
                else:
                    x_txt_dense = self._linear_words1(x_txt_dense)
        
        x_leader_dense = x_img_dense
        ### Делаем наиболее подходящие текстовые ассоциации под картинку лидера
        index_chose_img = torch.argmax(y_what_card_leader_choose, dim=1)
        output = [x_leader_dense[num,ind,:].repeat(self._params['num_words'],1)
                  for num,ind in enumerate(index_chose_img)]
        x_leader_dense_repeat = torch.stack(output, 0)
        logits = self._distance2axis_words(x_leader_dense_repeat, x_txt_dense)
        
        ### Веса для слов (Гумбель) или классика
        if self._params['gumbel']:
            weights_words = gumbel_softmax(logits, hard = True)
        else:
            weights_words = self._softmax_words(logits)
        
        #########################################
        final_words = torch.mul(weights_words, x_txt_dense.permute(2,0,1)).permute(1,2,0).sum(dim = 1)
        final_words_repeat = final_words.repeat(self._params['num_cards'],1,1).permute(1,0,2)
        logits_final = self._distance2axis_final(
                final_words_repeat, x_leader_dense
            )
        return self._softmax_final(logits)
    