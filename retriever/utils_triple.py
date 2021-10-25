import json
import os
import numpy as np
import scipy
import random
import unicodedata
from tqdm import tqdm
import glob

import torch
import nltk
import pickle

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# define the triple types
Triple_Types = ['minie_t', 'minie_t_cor','stanford_t']

class GraphRetrieverConfig:

    def __init__(self,
                 example_limit: int,
                 task: str,
                 max_seq_length: int,
                 max_select_num: int,
                 max_para_num: int,
                 tfidf_limit: int,
                 train_batchsize: int,

                 train_file_path: str,
                 use_redundant: bool,
                 use_multiple_redundant: bool,
                 max_redundant_num: int,

                 dev_file_path: str,
                 beam: int,
                 min_select_num: int,
                 no_links: bool,
                 pruning_by_links: bool,
                 expand_links: bool,
                 eval_chunk: int,
                 tagme: bool,
                 topk: int,
                 db_save_path: str):

        # General
        self.example_limit = example_limit

        self.open = False

        self.task = task
        assert task in ['hotpot_distractor', 'hotpot_open',
                        'squad', 'nq',
                        None]

        if task == 'hotpot_open' or (train_file_path is None and task in ['squad', 'nq']):
            self.open = True

        self.max_seq_length = max_seq_length

        self.max_select_num = max_select_num

        self.max_para_num = max_para_num

        self.tfidf_limit = tfidf_limit
        assert self.tfidf_limit is None or type(self.tfidf_limit) == int
        self.train_batchsize = train_batchsize

        # Train
        self.train_file_path = train_file_path

        self.use_redundant = use_redundant

        self.use_multiple_redundant = use_multiple_redundant
        if self.use_multiple_redundant:
            self.use_redundant = True

        self.max_redundant_num = max_redundant_num
        assert self.max_redundant_num is None or self.max_redundant_num > 0 or not self.use_multiple_redundant

        # Eval
        self.dev_file_path = dev_file_path
        #assert self.train_file_path is not None or self.dev_file_path is not None or task is None

        self.beam = beam

        self.min_select_num = min_select_num
        assert self.min_select_num >= 1 and self.min_select_num <= self.max_select_num

        self.no_links = no_links

        self.pruning_by_links = pruning_by_links
        if self.no_links:
            self.pruning_by_links = False

        self.expand_links = expand_links
        if self.no_links:
            self.expand_links = False

        self.eval_chunk = eval_chunk

        self.tagme = tagme

        self.topk = topk

        self.db_save_path = db_save_path

    def __str__(self):
        configStr = '\n\n' \
                    '### RNN graph retriever configurations ###\n' \
                    '@@ General\n' \
                    '- Example limit: ' + str(self.example_limit) + '\n' \
                                                                    '- Task: ' + str(self.task) + '\n' \
                                                                                                  '- Open: ' + str(
            self.open) + '\n' \
                         '- Max seq length: ' + str(self.max_seq_length) + '\n' \
                                                                           '- Max select num: ' + str(
            self.max_select_num) + '\n' \
                                   '- Max paragraph num (including links): ' + str(self.max_para_num) + '\n' \
                                                                                                        '- Limit of the initial TF-IDF pool: ' + str(
            self.tfidf_limit) + '\n' \
                                '\n' \
                                '@@ Train\n' \
                                '- Train file path: ' + str(self.train_file_path) + '\n' \
                                                                                    '- Use redundant: ' + str(
            self.use_redundant) + '\n' \
                                  '- Use multiple redundant: ' + str(self.use_multiple_redundant) + '\n' \
                                                                                                    '- Max redundant num: ' + str(
            self.max_redundant_num) + '\n' \
                                      '\n' \
                                      '@@ Eval\n' \
                                      '- Dev file path: ' + str(self.dev_file_path) + '\n' \
                                                                                      '- Beam size: ' + str(
            self.beam) + '\n' \
                         '- Min select num: ' + str(self.min_select_num) + '\n' \
                                                                           '- No links: ' + str(self.no_links) + '\n' \
                                                                                                                 '- Pruning by links (and top 1): ' + str(
            self.pruning_by_links) + '\n' \
                                     '- Exapnd links (for NQ): ' + str(self.expand_links) + '\n' \
                                                                                            '- Eval chunk: ' + str(
            self.eval_chunk) + '\n' \
                               '- Tagme: ' + str(self.tagme) + '\n' \
                                                               '- Top K: ' + str(self.topk) + '\n' \
                                                                                              '- DB save path: ' + str(
            self.db_save_path) + '\n' \
                                 '#########################################\n'

        return configStr


class InputExample(object):

    def __init__(self, guid, q, qt, c, para_dic, s_g, r_g, all_r_g, all_paras):

        self.guid = guid
        self.question = q
        self.question_t=qt
        self.context = c
        self.all_linked_paras_dic = para_dic
        self.short_gold = s_g
        self.redundant_gold = r_g
        self.all_redundant_gold = all_r_g
        self.all_paras = all_paras

        # paragraph index -> title
        self.title_order = []

def normalize(text):
    """Resolve different type of unicode encodings / capitarization in HotpotQA data."""
    text = unicodedata.normalize('NFD', text)
    return text[0].capitalize() + text[1:]

def make_wiki_id(title, para_index):
    title_id = "{0}_{1}".format(normalize(title), para_index)
    return title_id

def getData_db(title):
    # db1 = SqliteDict(os.path.join('./wiki_paras', 'train_paras.db'), autocommit=False)
    # db2 = SqliteDict(os.path.join('./wiki_paras', 'eval_paras.db'), autocommit=False)
    # key1=make_wiki_id(title,0)
    # if key1 in db2:
    #     return db2[key1]
    # elif key1 in db1:
    #     return db1[key1]
    #
    # if title in db2:
    #     return db2[title]
    # elif title in db1:
    #     return db1[title]
    return None

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,  q_ts_num, p_ts_num, p_gts_num, qt_pts, para_contents, output_masks, num_paragraphs):

        self.q_ts_num = q_ts_num,
        self.p_ts_num = p_ts_num,
        self.p_gts_num=p_gts_num,
        self.qt_pts = qt_pts,
        self.para_contents=para_contents
        self.output_masks = output_masks
        self.num_paragraphs = num_paragraphs

def expand_links(context, all_linked_paras_dic, all_paras):
    for context_title in context:
        # Paragraphs from the same article
        raw_context_title = context_title.split('_')[0]

        if context_title not in all_linked_paras_dic:
            all_linked_paras_dic[context_title] = {}

        for title in all_paras:
            if title == context_title or title in all_linked_paras_dic[context_title]:
                continue
            raw_title = title.split('_')[0]
            if raw_title == raw_context_title:
                all_linked_paras_dic[context_title][title] = all_paras[title]

def getfiles(dirname):
    filenames=[]
    for dir,_,files in os.walk(dirname):
        for i in files:
            if '.pkl' in i:
                filenames.append(os.path.join(dir,i))
    return filenames

class DataProcessor:

    def get_train_examples(self, graph_retriever_config,file_name):

        examples = []

        assert graph_retriever_config.train_file_path is not None


        if os.path.isfile(file_name):
            examples += self._create_examples(file_name, graph_retriever_config, "train")

        elif os.path.isdir(file_name):
            file_list = getfiles(file_name)

            for file_name in file_list[:1]:
                examples += self._create_examples(file_name, graph_retriever_config, "train")

        assert len(examples) > 0
        return examples

    def get_dev_examples(self, graph_retriever_config):

        examples = []

        assert graph_retriever_config.dev_file_path is not None

        file_name = graph_retriever_config.dev_file_path

        if os.path.exists(file_name):
            examples += self._create_examples(file_name, graph_retriever_config, "dev")
        else:
            file_list = list(glob.glob(file_name + '*'))
            for file_name in file_list:
                examples += self._create_examples(file_name, graph_retriever_config, "dev")

        assert len(examples) > 0
        return examples

    '''
    Read training examples from a json file
    * file_name: the json file name
    * graph_retriever_config: the graph retriever's configuration
    * task: a task name like "hotpot_open"
    * set_type: "train" or "dev"
    '''

    def merge_triple(self, tlist):
        factlist = []
        for i in tlist:
            factlist.extend(i)
        factlist = list(set(factlist))

        return factlist

    def fetchTriple_text(self, tlist):
        '''
        :param tlist: the list of triples
        :return: the list of sentence which converted from the triple. (flatten)
        '''
        tsens = []
        for t in tlist:
            sen = t
            if type(t)==list:
               sen = ' '.join(t)
            assert type(sen)==str

            if len(sen)>0 and sen[-1] == '.':
                if sen[-2] == ' ':
                    if sen not in tsens:
                        tsens.append(sen)
                else:
                    sen = sen.replace('.', ' .')
                    if sen not in tsens:
                        tsens.append(sen)
            else:
                sen = sen + ' .'
                if sen not in tsens:
                    tsens.append(sen)
        return tsens

    def _create_examples(self, file_name, graph_retriever_config, set_type):

        task = graph_retriever_config.task
        with open(file_name,'rb')as f:
            jsn = pickle.load(f)

        examples = []

        '''
        Limit the number of examples used.
        This is mainly for sanity-chacking new settings.
        '''
        if graph_retriever_config.example_limit is not None:
            random.shuffle(jsn)
            jsn = sorted(jsn, key=lambda x: x['q_id'])
            jsn = jsn[:graph_retriever_config.example_limit]

        '''
        Find the mximum size of the initial context (links are not included)
        '''
        graph_retriever_config.max_context_size = 0

        logger.info('#### Loading examples... from {} ####'.format(file_name))
        for (_, data) in enumerate(tqdm(jsn, desc='Example')):

            guid = data['q_id']
            question = data['question']['text']
            question_triple = []
            for ttype in Triple_Types:
                if ttype in data['question']:
                    question_triple.extend(data['question'][ttype])
            question_triples = self.fetchTriple_text(question_triple)

            context = data['context']  # {context title: paragraph}
            all_linked_paras_dic = data['all_linked_paras_dic']  # {context title: {linked title: paragraph}}
            short_gold = data['short_gold']  # [title 1, title 2] (Both are gold)
            redundant_gold = data['redundant_gold']  # [title 1, title 2, title 3] ("title 1" is not gold)
            all_redundant_gold = data['all_redundant_gold']

            '''
            Limit the number of redundant examples
            '''
            all_redundant_gold = all_redundant_gold[:graph_retriever_config.max_redundant_num]

            '''
            Control the size of the initial TF-IDF retrieved paragraphs
            *** Training time: to take a blalance between TF-IDF-based and link-based negative examples ***
            '''
            if graph_retriever_config.tfidf_limit is not None:
                new_context = {}
                for title in context:
                    if len(new_context) == graph_retriever_config.tfidf_limit:
                        break
                    new_context[title] = context[title]
                context = new_context

            '''
            Use TagMe-based context at test time.
            '''
            if set_type == 'dev' and task == 'nq' and graph_retriever_config.tagme:
                assert 'tagged_context' in data

                '''
                Reformat "tagged_context" if needed (c.f. the "context" case above)
                '''
                if type(data['tagged_context']) == list:
                    tagged_context = {c[0]: c[1] for c in data['tagged_context']}
                    data['tagged_context'] = tagged_context

                '''
                Append valid paragraphs from "tagged_context" to "context"
                '''
                for tagged_title in data['tagged_context']:
                    tagged_text = data['tagged_context'][tagged_title]
                    if tagged_title not in context and tagged_title is not None and tagged_title.strip() != '' and tagged_text is not None and tagged_text.strip() != '':
                        context[tagged_title] = tagged_text

            '''
            Clean "context" by removing invalid paragraphs
            '''
            removed_keys = []
            for title in context:
                if title is None or title.strip() == '' or context[title] is None or context[title]['text'][
                    0].strip() == '':
                    removed_keys.append(title)
            for key in removed_keys:
                context.pop(key)

            if task in ['squad', 'nq'] and set_type == 'train':
                new_context = {}

                orig_title = list(context.keys())[0].split('_')[0]

                orig_titles = []
                other_titles = []

                for title in context:
                    title_ = title.split('_')[0]

                    if title_ == orig_title:
                        orig_titles.append(title)
                    else:
                        other_titles.append(title)

                orig_index = 0
                other_index = 0

                while orig_index < len(orig_titles) or other_index < len(other_titles):
                    if orig_index < len(orig_titles):
                        new_context[orig_titles[orig_index]] = context[orig_titles[orig_index]]
                        orig_index += 1

                    if other_index < len(other_titles):
                        new_context[other_titles[other_index]] = context[other_titles[other_index]]
                        other_index += 1

                context = new_context

            '''
            Convert link format
            '''
            new_all_linked_paras_dic = {}  # {context title: {linked title: paragraph}}

            all_linked_paras_dic = data[
                'all_linked_paras_dic']  # rebacca: modify. as the original version didnot assign a value to it      {linked_title: paragraph} or mixed
            all_linked_para_title_dic = data[
                'all_linked_para_title_dic']  # {context_title: [linked_title_1, linked_title_2, ...]}

            removed_keys = []
            tmp = {}

            for key in all_linked_paras_dic:
                if type(all_linked_paras_dic[key]) == dict:
                    removed_keys.append(key)
                    for linked_title in all_linked_paras_dic[key]:
                        if linked_title not in all_linked_paras_dic:
                            tmp[linked_title] = all_linked_paras_dic[key][linked_title]

                        if key in all_linked_para_title_dic:
                            all_linked_para_title_dic[key].append(linked_title)
                        else:
                            all_linked_para_title_dic[key] = [linked_title]

            for key in removed_keys:
                all_linked_paras_dic.pop(key)

            for key in tmp:
                if key not in all_linked_paras_dic:
                    all_linked_paras_dic[key] = tmp[key]

            for context_title in context:
                if context_title not in all_linked_para_title_dic:
                    continue

                new_entry = {}

                for linked_title in all_linked_para_title_dic[context_title]:
                    if linked_title not in all_linked_paras_dic:
                        continue
                    new_entry[linked_title] = all_linked_paras_dic[linked_title]

                if len(new_entry) > 0:
                    new_all_linked_paras_dic[context_title] = new_entry
            # contain the linked paras by title in 'context' for each instance.
            # but we only process paras in the 'context'.
            all_linked_paras_dic = new_all_linked_paras_dic

            if set_type == 'dev':
                '''
                Clean "all_linked_paras_dic" by removing invalid paragraphs
                '''
                for c in all_linked_paras_dic:
                    removed_keys = []
                    links = all_linked_paras_dic[c]
                    for title in links:
                        if title is None or title.strip() == '' or links[title] is None or type(links[title]) != str or \
                                links[title].strip() == '':
                            removed_keys.append(title)
                    for key in removed_keys:
                        links.pop(key)

                all_paras = {}
                for title in context:
                    all_paras[title] = context[title]

                    if not graph_retriever_config.open:
                        continue

                    if title not in all_linked_paras_dic:
                        continue
                    for title_ in all_linked_paras_dic[title]:
                        if title_ not in all_paras:
                            all_paras[title_] = all_linked_paras_dic[title][title_]
            else:
                all_paras = None

            if set_type == 'dev' and graph_retriever_config.expand_links:
                expand_links(context, all_linked_paras_dic, all_paras)

            if set_type == 'dev' and graph_retriever_config.no_links:
                all_linked_paras_dic = {}

            graph_retriever_config.max_context_size = max(graph_retriever_config.max_context_size, len(context))

            '''
            Ensure that all the gold paragraphs are included in "context"
            '''
            if set_type == 'train':
                for t in short_gold + redundant_gold:
                    assert t in context

            examples.append(InputExample(guid=guid,
                                         q=question,
                                         qt=question_triples,
                                         c=context,
                                         para_dic=all_linked_paras_dic,
                                         s_g=short_gold,
                                         r_g=redundant_gold,
                                         all_r_g=all_redundant_gold,
                                         all_paras=all_paras))

        if set_type == 'dev':
            examples = sorted(examples, key=lambda x: len(x.all_paras))
        logger.info('Done!')

        return examples

def tokenize_question(question, tokenizer, max=378):
    tokens_q = tokenizer.tokenize(question)
    # if len(tokens_q)>300:#max=378
    #     tokens_q=tokens_q[:int(max/3)]
    tokens_q = ['[CLS]'] + tokens_q + ['[SEP]']

    return tokens_q
import string
def cor_resolu_triple(sentences,title,process_imcomplete=False):
    newsentences=[]
    string_pun=string.punctuation
    for sentence in sentences:
        newsen=[]
        if type(sentence) == list:
            sentence=' '.join(sentence)
        #cor resolution by nltk, process daici.
        words = nltk.word_tokenize(sentence)
        pos_tags =nltk.pos_tag(words)
        for word, tag in pos_tags:
            if tag=='PRP' or tag=='PRP$' or tag=='WP':# He,she, it; who what
                newsen.append(title)
            else:
                newsen.append(word)
        newsen = ' '.join(newsen)

        # detect the incomplete entity representation.
        if process_imcomplete:
            tempa=title
            for i in string_pun:
                tempa=tempa.replace(i,'')
            tempb=newsen
            for i in string_pun:
                tempb=tempb.replace(i,'')
            inters=set(tempa.split(" ")).intersection(set(tempb.split(" ")))

            for x in inters:
                if x in newsen:
                   newsen.replace(x,title)
        # as the matching of entity can induce redundancy of entities in a triple.

        newsentences.append(newsen)
    return newsentences

def fetch_Groundt(paragraph,title):
    triples=[]
    assert type(paragraph)==dict

    if 'ground_triple' in paragraph:
        triples=paragraph['ground_triple']
        triples = cor_resolu_triple(triples, title)

        # triples.append(paragraph['text'])  # path retriever
        return triples
    return triples


def fetchT_para(paragraph, title,process=False):
    triples = []
    if type(paragraph) != dict:
        return triples

    if 'merge_triples' in paragraph:
        triples=paragraph['merge_triples']
    else:
        for ttype in Triple_Types:
            if ttype in paragraph:
                triples.extend(paragraph[ttype])

    if len(triples) == 0:# when there no extracted triples, we use the original sentences as a representation.
        if type(paragraph['text'])==list:
            triples.extend(paragraph['text'])
        else:
            triples.extend(paragraph['text'].split('.'))
        return triples

    # transfer all the triples to the corresponding sentences.
    if process:
        ts = []
        for t in triples:
            sen = ' '.join(t)
            if len(sen)>0 and sen[-1] == '.':
                if sen[-2] == ' ':
                    if sen not in ts:
                        ts.append(sen)
                else:
                    sen = sen.replace('.', ' .')
                    if sen not in ts:
                        ts.append(sen)
            else:
                sen = sen + ' .'
                if sen not in ts:
                    ts.append(sen)
        triples=ts
    #make correfrence resolution for the triples extracted from the document.

    triples = cor_resolu_triple(triples,title)

    #triples.append(paragraph['text'])  # path retriever
    return triples

def encode_paras(paras,tokenizer,max_triple_length):
    '''
    :param paras: for the open-link paras, without triples. so we only encode the original text.
    :param tokenizer:
    :param max_triple_length: 256
    :return: the encoded id sequence for the paras.
    '''
    if type(paras)==str:# if the para is a long sentence.
        tokenizedi = tokenizer.tokenize(paras)
        tokenizedi = ['[CLS]'] + tokenizedi + ['[SEP]']
        paras_ids = (tokenizer.convert_tokens_to_ids(tokenizedi))

        if len(paras_ids) > max_triple_length:
            paras_ids = paras_ids[:max_triple_length]
        else:
            padding = [0] * (max_triple_length - len(paras_ids))
            paras_ids += padding
        return [paras_ids]
    elif type(paras)==list:# if the para is a set of sentences.
        # encoded_output = []
        # for i in paras:
        #     tokenizedi = tokenizer.tokenize(i)
        #     tokenizedi = ['[CLS]'] + tokenizedi + ['[SEP]']
        #     triple_ids = (tokenizer.convert_tokens_to_ids(tokenizedi))
        #
        #     if len(triple_ids) > max_triple_length:
        #         triple_ids = triple_ids[:max_triple_length]
        #     else:
        #         padding = [0] * (max_triple_length - len(triple_ids))
        #         triple_ids += padding
        #     encoded_output.append(triple_ids)

        '''special for the paragraph encoder.'''
        full_para=' '.join(paras)
        tokenizedi=tokenizer.tokenize(full_para)
        tokenizedi=['[CLS]'] + tokenizedi + ['[SEP]']
        full_para_ids = (tokenizer.convert_tokens_to_ids(tokenizedi))
        if len(full_para_ids)>max_triple_length:
            full_para_ids=full_para_ids[:max_triple_length]
        else:
            padding=[0]*(max_triple_length-len(full_para_ids))
            full_para_ids+=padding
        #encoded_output.append(full_para_ids)

        return [full_para_ids]

def merge_triple_sen(triples):
    '''
    merge the triples: if one triple can be seen as a part of another triple, it is a redundancy.
    we will remove the sub triple from the whole triple sets.
    '''
    remove=[]
    for i in range(len(triples)):
        for j in range(i+1, len(triples)):
            a=triples[i]
            alist=a.split(' ')
            b=triples[j]
            blist=b.split(' ')
            if set(alist).issubset(set(blist)):
                remove.append(a)
            elif set(blist).issubset(set(alist)):
                remove.append(b)
    #print('before : %d, after: %d'%(len(triples),len(list(set(triples)-set(remove)))))
    return list(set(triples)-set(remove))

def encode_triples(triples, tokenizer, max_triple_length=256):
    '''
    encode triples, for the special token comes from extraction tool, we remove it. and to reduce the redundancy over triples, we merge the triples.
    '''
    encoded_output = []
    triples=remove_special_word(triples)
    triples=merge_triple_sen(triples)

    for i in triples:
        if i=='':
            continue
        if type(i)==list:
            i = ' '.join(i)

        tokenizedi = tokenizer.tokenize(i)
        tokenizedi = ['[CLS]'] + tokenizedi + ['[SEP]']
        triple_ids = (tokenizer.convert_tokens_to_ids(tokenizedi))

        if len(triple_ids)>max_triple_length:
            triple_ids = triple_ids[:max_triple_length]
        else:
            padding = [0] * (max_triple_length - len(triple_ids))
            triple_ids += padding
        encoded_output.append(triple_ids)

    return encoded_output

def glove_encode(triples, wdict, merge="mean"):
    '''
    use glove embeds to encode the sentence words, and take the mean-vector as the sentence embedding representation.
    :param triples:
    :param wdict:
    :param merge:
    :return:
    '''
    encoded_output = []
    if merge == "mean":
        for sen in triples:
            words = sen.split(' ')
            sen_embed = []
            for word in words:
                if word in wdict:
                    sen_embed.append(np.asarray(wdict[word], "float32"))
                else:
                    sen_embed.append(np.asarray(np.zeros(300), "float32"))
            sen_embed = np.mean(sen_embed, axis=0)
            encoded_output.append(sen_embed)

    return encoded_output

class Para_features():
    def __init__(self,  q_embeds, para_embeds, lable):
        self.q_embed = q_embeds
        self.p_embeds = para_embeds
        self.label=lable

def convert_examples_to_features_text(examples,max_seq_length, tokenizer,max_size=50):
    features = []
    logger.info('#### Converting examples to features... ####')

    for (ex_index, example) in enumerate(tqdm(examples, desc='Example')):
        tokens_q = tokenize_question(example.question, tokenizer, max_seq_length)
        question_embeds = tokenizer.convert_tokens_to_ids(tokens_q)
        if len(question_embeds) > max_seq_length:
            question_embeds = question_embeds[:max_seq_length]
        else:
            padding_qe = [0] * (max_seq_length - len(question_embeds))
            question_embeds += padding_qe

        titles_list = example.short_gold + list(example.context.keys())
        label=[]
        para_embeds=[]
        for p in titles_list:
            title = p
            p = example.context[title]  # now, p refers to the context, while the former p refers to the title.
            p_text=p['text']
            if type(p_text)==list:
                p_text='.'.join(p_text)
            p_embeds = encode_paras(p_text, tokenizer, max_seq_length)
            if title in example.short_gold:
                label.append(1)
            else:
                label.append(0)
            para_embeds.append(p_embeds)
        if len(para_embeds)<max_size:
            label += (max_size - len(para_embeds)) * [0]
            para_embeds+=(max_size-len(para_embeds))*[[0]*max_seq_length]


        features.append(Para_features(q_embeds=question_embeds,
                                      para_embeds=para_embeds,
                                      lable=label))
    return features

def remove_special_word(sens):
    '''
    the triple extracted by tool, have some special tokens: QUANT_O_1, QUANT_R_1, QUANT_S_1 (which refer to number.)
    '''
    newsens=[]
    for sen in sens:
        words=sen.split(' ')
        newwords=[]
        for word in words:
            if 'QUANT_' in word:
                continue
            else:
                newwords.append(word)
        newsens.append(' '.join(newwords))
    return newsens

def convert_examples_to_features(examples, max_triples_size, max_seq_length, max_para_num,
                                      graph_retriever_config, tokenizer, train=False):
    """Loads a data file into a list of `InputBatch`s."""

    if not train and graph_retriever_config.db_save_path is not None:
        max_para_num = graph_retriever_config.max_context_size
        graph_retriever_config.max_para_num = max(graph_retriever_config.max_para_num, max_para_num)

    max_steps = graph_retriever_config.max_select_num  # 4 how many paragraphs will be choosen for each question.

    DUMMY = [0] * max_seq_length  # max_seq_length refers to the max length for each paragraphs.
    features = []

    logger.info('#### Converting examples to features... ####')

    for (ex_index, example) in enumerate(tqdm(examples, desc='Example')):
        tokens_q = tokenize_question(example.question, tokenizer, max_seq_length)

        q_triples = encode_triples(example.question_t, tokenizer, max_seq_length)
        question_embeds = tokenizer.convert_tokens_to_ids(tokens_q)
        if len(question_embeds)>max_seq_length:
            question_embeds = question_embeds[:max_seq_length]
        else:
            padding_qe = [0]*(max_seq_length-len(question_embeds))
            question_embeds+=padding_qe
        q_triples.append(question_embeds)# to avoid the question triple size=0, add the original question as a special triple.
        num_qts = len(q_triples)
        num_pts = []


        ##############
        # Short gold #
        ##############
        qt_pt_input = []
        title2index = {}

        # Append gold and non-gold paragraphs from context
        if train and graph_retriever_config.use_redundant and len(example.redundant_gold) > 0:
            if graph_retriever_config.use_multiple_redundant:
                titles_list = example.short_gold + [redundant[0] for redundant in example.all_redundant_gold] + list(
                    example.context.keys())
            else:
                titles_list = example.short_gold + [example.redundant_gold[0]] + list(example.context.keys())
        else:
            titles_list = example.short_gold + list(example.context.keys())

        for p in titles_list:  # title list store all the candidate paragraphs' title, include the ground truths and also the negatives.

            if len(qt_pt_input) == max_para_num:  # 50, set the maximum candidates are 50. it must contain the grounds by the line 497/499/501
                break

            # Avoid appending gold paragraphs as negative
            if p in title2index:
                continue

            # fullwiki eval
            # Gold paragraphs are not always in context
            if not train and graph_retriever_config.open and p not in example.context:
                continue

            title2index[p] = len(title2index)
            example.title_order.append(p)

            title = p
            p = example.context[title]  # now, p refers to the context, while the former p refers to the title.

            triples = fetchT_para(p, title)#
            p_triples = encode_triples(triples, tokenizer, max_seq_length)

            triples_special = encode_paras(p['text'],tokenizer,max_seq_length)#-1 is the paragraph text.
            p_triples.extend(triples_special)

            num_pts_ = len(p_triples)
            num_pts.append(num_pts_)

            triple_input_ids = q_triples + p_triples
            if len(triple_input_ids) < max_triples_size:
                for lack in range(max_triples_size - len(triple_input_ids)):
                    triple_input_ids.append([0] * max_seq_length)

            triple_input_idsx = scipy.sparse.csc_matrix(np.array(triple_input_ids[:max_triples_size]))
            qt_pt_input.append(triple_input_idsx)



        # Open-domain setting
        if graph_retriever_config.open:
            num_paragraphs_no_links = len(qt_pt_input)

            for p_ in example.context:
                if len(qt_pt_input) == max_para_num:
                    break

                if p_ not in example.all_linked_paras_dic:
                    continue
                # context contains the first hop, and then use the hyperlinked paragraphs for the next hop expansion.
                for l in example.all_linked_paras_dic[p_]:
                    if len(qt_pt_input) == max_para_num:
                        break

                    if l in title2index:
                        continue


                    # doc = getData_db(l)
                    # if doc==None:
                    p = example.all_linked_paras_dic[p_][l]
                    # else:
                    #    p = doc

                    if p==None or len(p)==0:
                        continue

                    triples = fetchT_para(p, title, process=True)
                    if len(triples) == 0 :
                        p_triples = encode_paras(p,tokenizer, max_seq_length)
                    else:
                        p_triples = encode_triples(triples, tokenizer, max_seq_length)
                    num_pts_ = len(p_triples)
                    num_pts.append(num_pts_)

                    triple_input_ids = q_triples + p_triples
                    if len(triple_input_ids) < max_triples_size:
                        for lack in range(max_triples_size - len(triple_input_ids)):
                            triple_input_ids.append([0] * max_seq_length)

                    triple_input_ids = scipy.sparse.csc_matrix(np.array(triple_input_ids[:max_triples_size]))

                    qt_pt_input.append(triple_input_ids)
                    title2index[l] = len(title2index)
                    example.title_order.append(l)


        assert len(qt_pt_input) <= max_para_num

        num_paragraphs = len(qt_pt_input)
        num_steps = len(example.short_gold) + 1  # 1 for EOE

        if train:
            assert num_steps <= max_steps
        #
        output_masks = [([1.0] * len(qt_pt_input) + [0.0] * (max_para_num - len(qt_pt_input) + 1)) for _ in
                        range(max_para_num + 2)]


        for i in range(len(qt_pt_input)):
            output_masks[i + 1][i] = 0.0

        if train:
            size = num_steps - 1

            for i in range(size):
                for j in range(size):
                    if i != j:
                        output_masks[i][j] = 0.0

            for i in range(size):
                output_masks[size][i] = 0.0

            for i in range(max_steps):
                if i > size:
                    for j in range(len(output_masks[i])):
                        output_masks[i][j] = 0.0

            # Use REDUNDANT setting
            # Avoid treating the redundant paragraph as a negative example at the first step
            if graph_retriever_config.use_redundant and len(example.redundant_gold) > 0:
                if graph_retriever_config.use_multiple_redundant:
                    for redundant in example.all_redundant_gold:
                        output_masks[0][title2index[redundant[0]]] = 0.0
                else:
                    output_masks[0][title2index[example.redundant_gold[0]]] = 0.0


        DUMMY_t = [0]*max_seq_length
        padding_triple = [DUMMY_t]*max_triples_size
        for lack in range(max_para_num - len(qt_pt_input)):
            qt_pt_input.append(scipy.sparse.csc_matrix(np.array(padding_triple)))
        # sparse matrix
        num_pts += [0]*(max_para_num-len(num_pts))

        if len(num_pts) < max_para_num:
            num_pts += [0] * (max_para_num - len(num_pts))

        features.append(
            InputFeatures(q_ts_num=num_qts,
                          p_ts_num=num_pts,
                          qt_pts=qt_pt_input,
                          output_masks=output_masks,
                          num_paragraphs=num_paragraphs,
                          num_steps=num_steps,
                          ex_index=ex_index))

        if not train or not graph_retriever_config.use_redundant or len(example.redundant_gold) == 0:
            continue

        ##################
        # Redundant gold #
        ##################

        for redundant_gold in example.all_redundant_gold:
            hist = set()
            qt_pt_input_r = []
            num_pts_r = []
#             num_qts_r = []

            # Append gold and non-gold paragraphs from context
            for p in redundant_gold + list(example.context.keys()):

                if len(qt_pt_input_r) == max_para_num:
                    break

                # assert p in title2index
                if p not in title2index:
                    # assert p not in redundant_gold
                    print('some error when process redundant_gold')
                    continue

                if p in hist:
                    continue
                hist.add(p)

                index = title2index[p]
                qt_pt_input_r.append(qt_pt_input[index])
                num_pts_r.append(num_pts[index])


            # Open-domain setting (mainly for HotpotQA fullwiki)
            if graph_retriever_config.open:

                for p in title2index:

                    if len(qt_pt_input_r) == max_para_num:
                        break

                    if p in hist:
                        continue
                    hist.add(p)

                    index = title2index[p]
                    qt_pt_input_r.append(qt_pt_input[index])
                    num_pts_r.append(num_pts[index])
#                     try:
#                          num_qts_r.append(num_qts[index])
#                     except:
#                         print(index)
#                         print(len(num_qts))

            assert len(qt_pt_input_r) <= max_para_num

            num_paragraphs_r = len(qt_pt_input_r)
            num_steps_r = len(redundant_gold) + 1

            assert num_steps_r <= max_steps

            output_masks_r = [([1.0] * len(qt_pt_input_r) + [0.0] * (max_para_num - len(qt_pt_input_r) + 1)) for _ in
                              range(max_para_num + 2)]

            size = num_steps_r - 1

            for i in range(size):
                for j in range(size):
                    if i != j:
                        output_masks_r[i][j] = 0.0

                if i > 0:
                    output_masks_r[i][0] = 1.0

            for i in range(size):  # size-1
                output_masks_r[size][i] = 0.0

            for i in range(max_steps):
                if i > size:
                    for j in range(len(output_masks_r[i])):
                        output_masks_r[i][j] = 0.0

            padding_triple = [DUMMY_t] * (max_para_num - len(qt_pt_input))
            qt_pt_input_r += padding_triple
            num_pts_r += [0] * (max_para_num - len(num_pts))


            if len(num_pts) < max_para_num:
                num_pts_r += [0] * (max_para_num - len(num_pts_r))

            features.append(
                InputFeatures(q_ts_num=num_qts,
                              p_ts_num=num_pts_r,
                              qt_pts=qt_pt_input_r,
                              output_masks=output_masks_r,
                              num_paragraphs=num_paragraphs_r,
                              num_steps=num_steps_r,
                              ex_index=None))

            if not graph_retriever_config.use_multiple_redundant:
                break

    logger.info('Done!')
    return features

def save_feature2dir(data_features, dirname, split = 1):
    if os.path.exists(dirname):
        print('save the converted data to %s.'%dirname)
    else:
        os.mkdir(dirname)

    #if split==1:
    filename = os.path.join(dirname, "data_feature_file_" + str(split) + ".npy")
    np.save(filename, data_features)
    print('finish write the numpy data to npy file %s.' % filename)
    return
    # totalnum = len(data_features)
    # avesize = int(totalnum/split)
    # for i in range(split):
    #     start = i*avesize
    #     if (i+1)*avesize +avesize >= len(data_features):
    #         end = len(data_features)
    #     else:
    #         end = min(len(data_features), i*avesize+avesize)
    #     data = data_features[start:end]
    #     filename = os.path.join(dirname, "data_feature_file_"+str(i)+".npy")
    #     np.save(filename,data)
    #     print('finish write the numpy data to npy file %s.'%filename)
    return

def save(model, output_dir, suffix):
    logger.info('Saving the checkpoint...')
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model_" + suffix + ".bin")

    status = True
    try:
        torch.save(model_to_save.state_dict(), output_model_file)
    except:
        status = output_model_file

    if status:
        logger.info('Successfully saved!')
    else:
        logger.warn('Failed!')

    return output_model_file

def load(output_dir, suffix):
    file_name = 'pytorch_model_' + suffix + '.bin'
    output_model_file = os.path.join(output_dir, file_name)
    return torch.load(output_model_file)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x
