import json
import nltk
import logging
import pickle
from tqdm import tqdm
import argparse
import string
import os
import numpy as np
import pickle
import faiss

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO

                        )

logger = logging.getLogger(__name__)


class Triple_cluster():
    def __init__(self,glove_dict='glove.json'):
        self.name='triple cluster'
        self.glove = glove_dict
        self.triple_types = ['stanford_t', 'minie_t_cor', 'minie_t']
        self.text_types = ['text', 'cor_text']
        if os.path.exists(self.glove):
            self.glove = self.read(glove_dict)
            print('finish load glove data with %d words.' % len(self.glove))

    def read(self,filename):
        logger.info('load data from ' + filename)
        if '.json' in filename:
            with open(filename, 'r')as f:
                data = json.load(f)
            return data
        elif '.pkl' in filename:
            with open(filename, 'rb')as f:
                data = pickle.load(f)
            return data
        elif '.txt' in filename:
            with open(filename, 'r')as f:
                data = f.readlines()
            return data
        else:
            print('cannot read data from the file!')
            return
        return

    def triple2sen(self,tlists):
        '''
        transfer the para's triples to flatternd sentence.
        :param tlists:
        :return:
        '''
        tsens=[]
        for t in tlists:
            sen = ' '.join(t)
            tsens.append(sen)
        return tsens

    def glove_encode(self, triples, merge="mean",oov=True):
        encoded_output = []
        if merge == "mean":
            for sen in triples:
                words = sen.split(' ')
                sen_embed = []
                for word in words:
                    if word in self.glove:
                        sen_embed.append(np.asarray(self.glove[word], "float32"))
                    else:
                        if oov:
                            sen_embed.append(np.asarray(np.zeros(300), "float32"))
                        else:
                            sen_embed.append(np.asarray(np.ones(300), "float32"))
                sen_embed = np.mean(sen_embed, axis=0)
                encoded_output.append(sen_embed)
        return encoded_output

    def sibling_concatenate(self,triples, maxsize=16, normalize=True):
        '''merge the two most similar triple to one triple.'''
        triples_sens = self.triple2sen(triples)
        glove_rep_tss = self.glove_encode(triples_sens, merge="mean", oov=True)
        glove_rep_tss = np.array(glove_rep_tss)
        features = glove_rep_tss
        # normalize the feature. You can replace "features" to your own one.
        if normalize:
            row_sums = np.linalg.norm(features, axis=1)
            features = features / row_sums[:, np.newaxis]

        n, dim = features.shape[0], features.shape[1]
        # index = faiss.IndexFlatIP(dim)
        index_flat = faiss.IndexFlatL2(dim)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 3, index_flat)
        index.add(features)
        topk = 1
        distances, indices = index.search(features, topk + 1)
        del index
        del res
        del index_flat
        out_of_bound = maxsize - len(triples)
        if out_of_bound > 16:
            out_of_bound = out_of_bound * 2
        else:
            out_of_bound = out_of_bound + 1

        score = (2 - distances) / 2
        score = score[:, 1]
        out_score = score[np.argpartition(score, out_of_bound)]
        nowscore = (out_score[-out_of_bound])

        newtriple = []
        already = []
        for idx, ind in enumerate(indices):
            if ind[0] not in already and score[idx] >= nowscore:
                combine = (triples[idx][:-1].split(' ')) + (triples[ind[1]].split(' '))
                combine = list(dict.fromkeys(combine))  # remove redundant elements.
                newtriple.append(' '.join(combine))
                already.append(ind[1])
                already.append(ind[0])
            elif ind[0] not in already:
                newtriple.append(triples[idx])

        return newtriple

    def s_canopiese(self,triples):
        '''
        split the triples into smaller canopies by same subject.
        :param triples:
        :return:
        '''
        canopies=dict()
        for t in triples:
            subject=t[0]
            if subject not in canopies:
                canopies[subject]=[t]
            else:
                canopies[subject]=canopies[subject]+[t]
        return canopies

    def sp_canopiese(self,triples):
        '''
        split the triples into smaller canopies by same subject-predicate.
        :param triples:
        :return:
        '''
        canopies=dict()
        for t in triples:
            subject=t[0]
            predicate=t[1]
            sp=subject+'$$'+predicate
            if sp not in canopies:
                canopies[sp]=[t]
            else:
                canopies[sp]=canopies[sp]+[t]
        return canopies





