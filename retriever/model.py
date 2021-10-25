from transformers import BertPreTrainedModel, BertModel
from scipy import spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import time
from torch.nn.parameter import Parameter
from utils_triple import tokenize_question
import torch.nn as nn
from utils_triple import expand_links
import numpy as np


class BertForGraphRetriever(BertPreTrainedModel):

    def __init__(self, config, para_encoder_model, graph_retriever_config):
        super(BertForGraphRetriever, self).__init__(config)
        self.para_encoder_model = para_encoder_model
        self.graph_retriever_config = graph_retriever_config
        config.output_hidden_states = True

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initial state
        self.s = Parameter(torch.FloatTensor(config.hidden_size).uniform_(-0.1, 0.1))

        # Scaling factor for weight norm
        self.g = Parameter(torch.FloatTensor(1).fill_(1.0))

        # triple weight
        self.ta = Parameter(torch.FloatTensor(1).fill_(0.1))

        self.w1 = Parameter(torch.FloatTensor(1).fill_(0.1))
        self.w2 = Parameter(torch.FloatTensor(1).fill_(0.1))
        # self.updatew = Parameter(torch.FloatTensor(1).fill_(0.1))
        self.attention_matrix = torch.FloatTensor(graph_retriever_config.train_batchsize,
                                                  graph_retriever_config.max_select_num,
                                                  graph_retriever_config.max_para_num + 1).fill_(0.1)

        # EOE and output bias
        self.eos = Parameter(torch.FloatTensor(config.hidden_size).uniform_(-0.1, 0.1))
        self.bias = Parameter(torch.FloatTensor(1).zero_())

        self.apply(self._init_weights)
        self.cpu = torch.device('cpu')

    '''
    state: (B, 1, D)
    '''

    def weight_norm(self, state):
        state = state / state.norm(dim=2).unsqueeze(2)
        state = self.g * state
        return state

    '''
    input_ids, token_type_ids, attention_mask: (B, N, L)
    B: batch size
    N: maximum number of Q-P pairs
    L: maximum number of input tokens
    '''

    def encode(self, input_ids, token_type_ids, attention_mask, split_chunk=None):
        B = input_ids.size(0)
        N = input_ids.size(1)
        L = input_ids.size(2)
        input_ids = input_ids.contiguous().view(B * N, L)
        token_type_ids = token_type_ids.contiguous().view(B * N, L)
        attention_mask = attention_mask.contiguous().view(B * N, L)

        # [CLS] vectors for Q-P pairs
        if split_chunk is None:
            # input_ids represent the question-paragraph list;
            encoded_layers = self.bert(input_ids, token_type_ids, attention_mask)
            pooled_output = encoded_layers[0][:, 0]

        # an option to reduce GPU memory consumption at eval time, by splitting all the Q-P pairs into smaller chunks
        else:
            assert type(split_chunk) == int

            TOTAL = input_ids.size(0)
            start = 0

            while start < TOTAL:
                end = min(start + split_chunk - 1, TOTAL - 1)
                chunk_len = end - start + 1

                input_ids_ = input_ids[start:start + chunk_len, :]
                token_type_ids_ = token_type_ids[start:start + chunk_len, :]
                attention_mask_ = attention_mask[start:start + chunk_len, :]

                encoded_layers = self.bert(input_ids_, token_type_ids_, attention_mask_)
                encoded_layers = encoded_layers[0][:, 0]

                if start == 0:
                    pooled_output = encoded_layers
                else:
                    pooled_output = torch.cat((pooled_output, encoded_layers), dim=0)

                start = end + 1

            pooled_output = pooled_output.contiguous()

        paragraphs = pooled_output.view(pooled_output.size(0) // N, N, pooled_output.size(1))  # (B, N, D), D: BERT dim
        EOE = self.eos.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        EOE = EOE.expand(paragraphs.size(0), EOE.size(1), EOE.size(2))  # (B, 1, D)
        EOE = self.bert.encoder.layer[-1].output.LayerNorm(EOE)
        paragraphs = torch.cat((paragraphs, EOE), dim=1)  # (B, N+1, D)

        # Initial state
        state = self.s.expand(paragraphs.size(0), 1, self.s.size(0))  # (B, 1, D)
        state = self.weight_norm(state)

        return paragraphs, state

    def updated_query(self, query_triples, updated):
        query_triples = query_triples.long()
        updated = updated.long()
        return torch.cat((query_triples, updated), dim=0)

    def predicate(self, input_qt_pt, input_content, input_pt_nums, input_qt_nums,input_gt_nums,
                 train_strategy='triple_only',metrics='max'):
        if train_strategy == 'triple_only':
            output_triple = self.triple_forward_single_based_only_triple(input_qt_pt, input_content, input_pt_nums,
                                                                         input_qt_nums, input_gt_nums,

                                                                         calculation=metrics,eval=True)
        elif train_strategy == 'content_weighted_sum' and self.para_encoder_model != None:
            output_triple = self.triple_forward_single_based_para_triple(input_qt_pt, input_content, input_pt_nums,
                                                                         input_qt_nums, input_gt_nums,eval=True
                                                                         )
        else:
            print('no train strategy!!!!!')
            if self.para_encoder_model==None:
                print('because the second one misss')
            return None

        output = output_triple #.to(self.bert.device)
        return output


    def forward(self, target, input_qt_pt, input_content, input_pt_nums, input_qt_nums, input_gt_nums,
                maximum_triple_size, merge_way, train_strategy='triple_only',metrics='max'):
        if train_strategy == 'triple_only':
            output_triple = self.triple_forward_single_based_only_triple(input_qt_pt, input_content, input_pt_nums,
                                                                         input_qt_nums, input_gt_nums,
                                                                         maximum_size=maximum_triple_size,
                                                                         calculation=metrics)
        elif train_strategy == 'content_weighted_sum' and self.para_encoder_model != None:
            output_triple = self.triple_forward_single_based_para_triple(input_qt_pt, input_content, input_pt_nums,
                                                                         input_qt_nums, input_gt_nums
                                                                         )
        else:
            print('no train strategy!!!!!')
            if self.para_encoder_model==None:
                print('because the second one misss')
            return None

        output = output_triple.to(self.bert.device)
        loss = F.binary_cross_entropy_with_logits(output, target, reduction='mean')
        return loss

    def triple_forward_single_based_para_triple(self, input_qt_pt, input_content, input_pt_nums, input_qt_nums,
                                                input_gt_nums,calculation='max',eval=False):
        '''
        train on single para, we now just finetune the bert encoder to learn the similar q-p triples embedding closer.
        :param input_qt_pt:
        :param input_pt_nums:
        :param input_qt_nums:
        :param max_num_steps:
        :return:
        '''

        B = input_qt_pt.size(0)
        N = input_qt_pt.size(1)
        attention_matrix = torch.zeros(B, N)


        for ii in range(B):
            question_triple = input_qt_pt[ii, 0, :(input_qt_nums[ii].item()), :]
            if eval==True:
                with torch.no_grad():
                    encoded_layers = self.bert(question_triple)
            else:
                encoded_layers = self.bert(question_triple)

            question_triple_embeds = encoded_layers[0][:,0]
            assert self.para_encoder_model!=None

            with torch.no_grad():
                encoded_layers=self.para_encoder_model.bert(question_triple)# question triple==question text
                question_text_embeds = encoded_layers[0][:, 0]# 1 768

            for j in range(N):
                if input_gt_nums!=-1:
                    ground_para_triples = input_qt_pt[ii, j,
                                      input_qt_nums[ii].item() + input_pt_nums[ii][j].item(): input_qt_nums[ii].item() +
                                                                                              input_pt_nums[ii][
                                                                                                  j].item() +
                                                                                              input_gt_nums[ii][
                                                                                                  j].item(), :]

                para_triples = input_qt_pt[ii, j,
                               input_qt_nums[ii].item():input_qt_nums[ii].item() + input_pt_nums[ii][j].item(), :]

                if ground_para_triples.size(0) > 0 and input_gt_nums!=-1:
                    if ground_para_triples.size(0) == 1:
                        try:
                            para_triples = torch.cat((ground_para_triples, para_triples), dim=0)
                        except:
                            ground_para_triples = torch.unsqueeze(ground_para_triples[-1, :], 0)
                            para_triples = torch.cat((ground_para_triples, para_triples), dim=0)
                    else:
                        para_triples = torch.cat((ground_para_triples, para_triples), dim=0)
                if para_triples.size(0) > 39:
                    para_triples = para_triples[:39, :]

                if eval == True:
                    with torch.no_grad():
                        encoded_layers = self.bert(para_triples)
                else:
                    encoded_layers = self.bert(para_triples)
                para_triple_embeds = encoded_layers[0][:, 0]

                s1, _, _ = self.triple_similarity(question_triple_embeds, para_triple_embeds)

                with torch.no_grad():
                    encoded_layers = self.para_encoder_model.bert(input_content[ii,j,:])  # question triple==question text
                    para_text_embeds = encoded_layers[0][:, 0]  # 1 768

                s2, _, _=self.triple_similarity(question_text_embeds,para_text_embeds)

                s1=s1.to(self.w1.device)
                s2=s2.to(self.w2.device)
                s= self.w1*s1 + self.w2*s2
                attention_matrix[ii][j]=s

        return attention_matrix

    def triple_forward_eval(self, input_qt_pt, input_qt_nums, input_pt_nums, N):
        '''
        rnn of a paragraph paths.
        '''
        beam = self.graph_retriever_config.beam
        attention_matrix = torch.zeros(1, N + 1)
        total_input_size = input_pt_nums.size(0)

        updated_query = None

        question_triple_embeds = input_qt_pt[:(input_qt_nums.item()), :]
        if input_qt_nums.item() < 1:
            attention_matrix[:][:] = 0
            attention_matrix = attention_matrix.expand(beam, attention_matrix.size(1))
            return attention_matrix, input_qt_pt  # beam, N+1

        retrieve_dict_q_index = dict()
        retrieve_dict_p = dict()
        score_dict = dict()

        for j in range(total_input_size):  # the total length
            para_triples = input_qt_pt[input_qt_nums.item():input_pt_nums[j].item(), :]
            s, max_q_triple_index, max_p_triple = self.triple_similarity(question_triple_embeds,
                                                                         para_triples)
            attention_matrix[0][j] = s
            if max_q_triple_index != None:
                score_dict[j] = s
                retrieve_dict_q_index[j] = max_q_triple_index
                retrieve_dict_p[j] = max_p_triple

        topk_paras_now = self.topk_para_triples(score_dict, topk=10,
                                                theta=0.1)  # store the top k paras (key_id, score) based on triple similarity
        topk_new_q_triples, updated = self.generate_newQt(topk_paras_now, retrieve_dict_q_index,
                                                          retrieve_dict_p, question_triple_embeds)

        attention_matrix = attention_matrix.expand(beam, attention_matrix.size(1))

        return attention_matrix, topk_new_q_triples

    def triple_similarity(self, q_ts, p_ts):
        '''q_ts: all the triples of question. N,768
           p_ts: all the triples of paragraphs. M,768

           output: the max similarity between the triple pair (q_t, p_t).
        '''
        a = q_ts
        b = p_ts
        eps = 1e-6
        N = q_ts.size(0)
        M = p_ts.size(0)

        if N == 0 or M == 0:
            return torch.tensor(0), None, None

        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))  # N*M
        s = torch.max(sim_mt)
        flatten_loc = torch.argmax(sim_mt)
        max_q_triple_index = flatten_loc // M  # the matched triple index in the question triple set.
        max_p_triple_index = flatten_loc % M
        return s, max_q_triple_index, max_p_triple_index

    def triple_similarity_mean(self, q_ts, p_ts):
        a = q_ts
        b = p_ts
        eps = 1e-6
        N = q_ts.size(0)
        M = p_ts.size(0)

        if N == 0 or M == 0:
            return torch.tensor(0), None, None

        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))  # N*M
        s = torch.max(sim_mt,1)#N*1
        print(sim_mt)
        print(sim_mt.size())
        print(s)
        s = s.values
        s = s.sum()
        return s

    def topk_para_triples(self, test, topk, theta=0.6):
        sd = sorted(test.items(), key=lambda test: (test[1], test[0]), reverse=True)
        sd = dict(sd)

        topk_paras = dict()
        for i in sd:
            if sd[i] > theta and len(topk_paras) < topk:
                topk_paras[i] = sd[i]
            elif sd[i] < theta:
                break
        return topk_paras

    def generate_newQt(self, topk_paras_now, retrieve_dict_q_index, retrieve_dict_p, q_triple_embed):

        new_q_embed = None
        update_loc = []
        updated = False

        for p in topk_paras_now:
            para_triple = retrieve_dict_p[p]  # the correponding matched triple in the paragraph.
            q_triple = q_triple_embed[retrieve_dict_q_index[p], :]
            update_loc.append(retrieve_dict_q_index[p])

            data = torch.cat((para_triple, q_triple), dim=0)
            new = self.rw_t(data)
            new = torch.unsqueeze(new, dim=0)
            if new_q_embed == None:
                new_q_embed = new
            else:
                new_q_embed = torch.cat((new_q_embed, new), dim=0)

            updated = True

        for idx in range(len(q_triple_embed)):
            if idx not in update_loc:
                data = q_triple_embed[idx, :]
                data = torch.unsqueeze(data, dim=0)
                if new_q_embed == None:
                    new_q_embed = (q_triple_embed[idx, :])
                else:
                    try:
                        new_q_embed = torch.cat((new_q_embed, data), dim=0)
                    except:
                        new_q_embed = torch.unsqueeze(new_q_embed, dim=0)
                        new_q_embed = torch.cat((new_q_embed, data), dim=0)
        return new_q_embed, updated

    def retriever_expanding(self, score_dict, s, key, q_index, previous):
        score_dict = score_dict
        if s != 0:
            last_con = previous[q_index]
            if last_con != '#question#':
                print('*' * 20)
                newkey = key + '###' + last_con
                if key in score_dict:
                    score_dict[newkey] = score_dict[last_con] + s
                else:
                    score_dict[newkey] = s
            else:
                '''have not been retrieved based on the query triples in the previous hop.
                '''
                #             print(score_dict[key])
                #             print(s)
                score_dict[key] = s
        else:
            return score_dict
        return score_dict

    def triple_forward_single_based_only_triple(self, input_qt_pt, input_content, input_pt_nums, input_qt_nums,
                                                input_gt_nums, calculation='max',eval=False):
        '''
        train on single para, we now just finetune the bert encoder to learn the similar q-p triples embedding closer.
        :param input_qt_pt:
        :param input_pt_nums:
        :param input_qt_nums:
        :param max_num_steps:
        :return:
        '''

        B = input_qt_pt.size(0)
        N = input_qt_pt.size(1)
        attention_matrix = torch.zeros(B, N)


        maximum_question_size = 4
        for ii in range(B):
            question_triple = input_qt_pt[ii, 0, :(input_qt_nums[ii].item()), :]
            if eval==True:
                with torch.no_grad():
                    encoded_layers = self.bert(question_triple)
            else:
                encoded_layers = self.bert(question_triple)
            question_triple_embeds = encoded_layers[0][:,
                                     0]

            if input_qt_nums[ii].item() < 1:
                attention_matrix[ii][:] = 0
                continue

            for j in range(N):  # calcluate the paragraph with max
                if input_gt_nums!=-1:
                    ground_para_triples = input_qt_pt[ii, j,
                                      input_qt_nums[ii].item() + input_pt_nums[ii][j].item(): input_qt_nums[ii].item() +
                                                                                              input_pt_nums[ii][j].item() +
                                                                                              input_gt_nums[ii][j].item(), :]

                para_triples = input_qt_pt[ii, j,
                               input_qt_nums[ii].item():input_qt_nums[ii].item() + input_pt_nums[ii][j].item(), :]
                if input_gt_nums!=-1 and input_gt_nums[ii][j].item()>0:
                    try:
                        para_triples = torch.cat((ground_para_triples, para_triples), dim=0)
                    except:
                        para_triples = torch.cat((torch.unsqueeze(ground_para_triples[-1, :], 0), para_triples), dim=0)
                if para_triples.size(0) > 39:
                    para_triples = para_triples[:39, :]
                if eval == True:
                    with torch.no_grad():
                        encoded_layers = self.bert(para_triples)
                else:
                    encoded_layers = self.bert(para_triples)
                para_triple_embeds = encoded_layers[0][:, 0]
                if calculation=='max':
                    s1, _, _ = self.triple_similarity(question_triple_embeds, para_triple_embeds)
                elif calculation=='mean':
                    s1, _, _ = self.triple_similarity_mean(question_triple_embeds,para_triple_embeds)
                attention_matrix[ii][j] = s1


        del para_triples
        del question_triple
        del question_triple_embeds
        del encoded_layers

        return attention_matrix

    # def predicate(self, input_qt_pt, input_pt_nums, input_qt_nums, maximum_size=36):
    #     '''
    #     input_qt_pt: [batch_size, 500, 36,512]
    #     '''
    #     B = input_qt_pt.size(0)
    #     N = input_qt_pt.size(1)
    #     bert_score_dict = np.zeros((B, N))
    #     score_dict = np.zeros((B, N))
    #     qt_index_matrix = np.zeros((B, N))
    #     pt_index_matrix = np.zeros((B, N))
    #
    #     for ii in range(B):
    #         question_triples = input_qt_pt[ii, 0, :(input_qt_nums[ii].item()), :]
    #         with torch.no_grad():
    #             encoded_layers = self.bert(question_triples)
    #         question_triple_embeds = encoded_layers[0][:, 0]
    #
    #         for j in range(N):
    #             para_triples = input_qt_pt[ii, j, input_qt_nums[ii].item():input_pt_nums[ii][j].item(), :]
    #             # if para_triples.size(0)>maximum_size-question_triples.size(0):
    #             #     #print('***triple size out of limitation.***')
    #             para_triples = para_triples[:maximum_size - question_triples.size(0), :]
    #             with torch.no_grad():
    #                 encoded_layers = self.bert(para_triples)
    #             para_triple_embeds = encoded_layers[0][:, 0]
    #             s1, qt_index, pt_index = self.triple_similarity(question_triple_embeds, para_triple_embeds)
    #             bert_score_dict[ii][j] = s1.item()
    #             s1 = s1.to(self.ta.device)
    #             true_score = self.ta * s1
    #             score_dict[ii][j] = true_score.item()
    #             if qt_index != None:
    #                 qt_index_matrix[ii][j] = qt_index.item()
    #             else:
    #                 qt_index_matrix[ii][j] = -1
    #             if pt_index != None:
    #                 pt_index_matrix[ii][j] = pt_index.item()
    #             else:
    #                 pt_index_matrix[ii][j] = -1
    #     return score_dict, bert_score_dict, qt_index_matrix, pt_index_matrix

    def triple_forward(self, input_qt_pt, input_para, input_pt_nums, input_qt_nums, max_num_steps):
        B = input_qt_pt.size(0)
        N = input_qt_pt.size(1)  # total size of question-para pairs in a batch.

        attention_matrix = torch.zeros(B, max_num_steps, N + 1)
        # attention_matrix = Variable(self.attention_matrix, requires_grad=False)
        for ii in range(B):
            question_triple_embeds = input_qt_pt[ii, 0, :(input_qt_nums[ii].item()), :]
            encoded_layers = self.bert(question_triple_embeds)
            question_triple_embeds = encoded_layers[0][:,
                                     0]  # 一个batch中 question triples 的embeddings: N, number of triples,768 (each triple can be seen as a sentence)

            para_triples = None
            retrieve_dict_q_index = dict()
            retrieve_dict_p = dict()
            score_dict = dict()

            for i in range(max_num_steps):
                if i == 0:
                    for j in range(N):  # calcluate the paragraph with max
                        if para_triples == None:
                            para_triples = input_qt_pt[ii, j, input_qt_nums[ii].item():input_pt_nums[ii][j].item(), :]
                        else:
                            para_triples = torch.cat((para_triples, input_qt_pt[ii, j,
                                                                    input_qt_nums[ii].item():input_pt_nums[ii][
                                                                        j].item(), :]), dim=0)

                    encoded_layers = self.bert(para_triples)
                    para_triple_embeds = encoded_layers[0][:, 0]
                    start = 0

                    for j in range(N):
                        end = start + input_pt_nums[ii][j].item()
                        # end = min(end, para_triples.size(0))
                        para_triple_embeds = para_triples[start:end, :]
                        start = end
                        if para_triple_embeds.size(0) == 0:
                            attention_matrix[ii][i][j] = 0
                            continue

                        s, max_q_triple_index, max_p_triple = self.triple_similarity(question_triple_embeds,
                                                                                     para_triple_embeds)  # select which triple to update query triples.
                        if max_q_triple_index != None:
                            score_dict[j] = s
                            retrieve_dict_q_index[j] = max_q_triple_index
                            retrieve_dict_p[j] = max_p_triple
                        attention_matrix[ii][i][j] = s

                    # one batch, for hop-i retriever: only the para with maximum triple-match score.
                    topk_paras_now = self.topk_para_triples(score_dict, topk=10,
                                                            theta=0.6)  # store the top k paras (key_id, score) based on triple similarity
                    topk_new_q_triples, updated = self.generate_newQt(topk_paras_now, retrieve_dict_q_index,
                                                                      retrieve_dict_p, question_triple_embeds)

                else:
                    if updated == False:
                        attention_matrix[ii][i][:] = attention_matrix[ii][i - 1][:]
                        continue
                    else:
                        score_dict = dict()
                        retrieve_dict_q_index = dict()
                        retrieve_dict_p = dict()
                        start = 0
                        for j in range(N):
                            '''use the hop i-1, the selected updated triples to calculate a score for hop i. 
                             the similarity between question triples and paras is calculated once when i=0.
                            '''
                            end = start + input_pt_nums[ii][j].item()
                            para_triple_embeds = para_triples[start:end, :]
                            s, q_index, dict_p = self.triple_similarity(topk_new_q_triples,
                                                                        para_triple_embeds)  # select which triple to update query triples.
                            if s != 0 and q_index != None and dict_p != None:
                                score_dict[j] = s
                                retrieve_dict_q_index[j] = q_index
                                retrieve_dict_p[j] = dict_p
                            attention_matrix[ii][i][j] = s
                        topk_paras_now = self.topk_para_triples(score_dict, topk=10, theta=0.6)
                        topk_new_q_triples, updated = self.generate_newQt(topk_paras_now, retrieve_dict_q_index,
                                                                          retrieve_dict_p, topk_new_q_triples)

        return attention_matrix[:B, :max_num_steps, :N + 1]
