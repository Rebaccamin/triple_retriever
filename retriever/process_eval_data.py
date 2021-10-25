import json

import os
# import moxing as mox
# # import psutil
#
# os.system("pip install tqdm")
# # os.system("pip install pytorch-pretrained-bert")
# os.system("pip install transformers")
# os.system("pip install torch")
# os.system("pip install scipy")
from tqdm import tqdm
import os
import numpy as np
import scipy
from scipy.constants import pi
import scipy.sparse as ss
print("sciPy - pi = %.16f" % pi)
import torch
from utils_triple import InputExample
from utils_triple import InputFeatures

# from graph_retriever.utils import tokenize_question
# from graph_retriever.utils import tokenize_paragraph
# from graph_retriever.utils import GraphRetrieverConfig
# from graph_retriever.utils import expand_links
# from graph_retriever.modeling_graph_retriever import BertForGraphRetriever
# import numpy as np
# from utils_triple import BertForGraphRetriever
# from utils_triple import InputExample, InputFeatures
from utils_triple import expand_links, tokenize_question
# from utils_triple import GraphRetrieverConfig
# # from graph_retriever_triple.utils_triple import warmup_linear
#
# from transformers import BertTokenizer
#
# from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
# change the download location for pretrained bert model
from pathlib import Path
import collections
import nltk

PYTORCH_PRETRAINED_BERT_CACHE = Path("/export/data/chengminu/.pytorch_pretrained_bert")

import logging
import pickle
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_triple(tlist):
    factlist = []
    for i in tlist:
        factlist.extend(i)
    factlist = list(set(factlist))

    return factlist

def fetchTriple_text(tlist):
    '''
    :param tlist: the list of triples
    :return: the list of sentence which converted from the triple. (flatten)
    '''
    tsens = []
    for t in tlist:
        if type(t)==list:
            sen = ' '.join(t)
        else:
            sen=t
        if len(sen)>0 and sen[-1] == '.':
            if sen[-2] == ' ':
                tsens.append(sen)
            else:
                sen = sen.replace('.', ' .')
            tsens.append(sen)
        elif len(sen)>0:
            sen = sen + ' .'
            tsens.append(sen)
    return tsens

def read(filename):
    if '.pkl' in filename:
        with open(filename, 'rb')as f:
            data = pickle.load(f)
        return data
    elif '.json' in filename:
        with open(filename, 'r')as f:
            data = json.load(f)
        return data
    elif '.txt' in filename:
        with open(filename, 'r') as f:
            data = f.readlines()
        return data
    else:
        print('we can not read the file: %s. ' % filename)

def getfiles(dirname):
    filenames = []
    for dir, _, files in os.walk(dirname):
        for file in files:
            filenames.append(os.path.join(dir, file))
    logger.info('Load %d files from the dir:%s.' % (len(filenames), dirname))
    return filenames

def create_examples(files, text_type='text'):
    # task = graph_retriever_config.task

    examples = []
    '''
    Find the mximum size of the initial context (links are not included)
    '''
    max_context_size = 0

    for file in tqdm(files):
        data=read(file)
        guid = data['q_id']
        if text_type=='text':
            question = data['question']['text']
        else:
            question=data['question']['cor_text']
        question_triple = data['question']['merged_triples']
        question_triples = fetchTriple_text(question_triple)

        context = data['context']

        all_linked_paras_dic = {}  # {context title: {linked title: paragraph}}
        '''
        Clean "context" by removing invalid paragraphs
        '''
        removed_keys = []
        for title in context:
            if title is None or title.strip() == '' or context[title]['text'] is None or context[title][
                'text'].strip() == '':
                removed_keys.append(title)
        for key in removed_keys:
            context.pop(key)
        all_paras = {}
        for title in context:
            all_paras[title] = context[title]

        # if graph_retriever_config.expand_links:  # when do evaluation, we donot use the linked url.
        #     expand_links(context, all_linked_paras_dic, all_paras)

        max_context_size = max(500, len(context))

        examples.append(InputExample(guid=guid,
                                     q=question,
                                     qt=question_triples,
                                     c=context,
                                     para_dic=all_linked_paras_dic,
                                     s_g=None, r_g=None, all_r_g=None,
                                     all_paras=all_paras))

    print(max_context_size)#500
    return examples

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

def encode_triples(triples, tokenizer, max_triple_length):
    '''
       encode triples, for the special token comes from extraction tool, we remove it. and to reduce the redundancy over triples, we merge the triples.
       '''
    encoded_output = []
    triples = remove_special_word(triples)
    triples = merge_triple_sen(triples)

    for i in triples:
        if i == '':
            continue
        if type(i) == list:
            i = ' '.join(i)

        tokenizedi = tokenizer.tokenize(i)
        tokenizedi = ['[CLS]'] + tokenizedi + ['[SEP]']
        triple_ids = (tokenizer.convert_tokens_to_ids(tokenizedi))

        if len(triple_ids) > max_triple_length:
            triple_ids = triple_ids[:max_triple_length]
        else:
            padding = [0] * (max_triple_length - len(triple_ids))
            triple_ids += padding
        encoded_output.append(triple_ids)

    return encoded_output

import string
def cor_resolu_triple(sentences,title):
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
            if tag=='PRP' or tag=='WP':# He,she, it; who what
                newsen.append(title)
            else:
                newsen.append(word)
        newsen = ' '.join(newsen)

        # detect the incomplete entity representation.
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

        #sentence = newsen.lower()
        newsentences.append(newsen)
    return newsentences


def fetchT_eval_para(paragraph,title):
    triples = []
    if type(paragraph) != dict:
        return triples

    if 'merge_triples' in paragraph:
        triples = paragraph['merge_triples']
    else:
        Triple_Types=['stanford_t']
        for ttype in Triple_Types:
            if ttype in paragraph:
                triples.extend(paragraph[ttype])

    if len(triples) == 0:  # when there no extracted triples, we use the original sentences as a representation.
        if type(paragraph['text']) == list:
            triples.extend(paragraph['text'])
        else:
            triples.extend(paragraph['text'].split('.'))
        return triples

    # transfer all the triples to the corresponding sentences.
    # if process:
    #     ts = []
    #     for t in triples:
    #         sen = ' '.join(t)
    #         if len(sen) > 0 and sen[-1] == '.':
    #             if sen[-2] == ' ':
    #                 if sen not in ts:
    #                     ts.append(sen)
    #             else:
    #                 sen = sen.replace('.', ' .')
    #                 if sen not in ts:
    #                     ts.append(sen)
    #         else:
    #             sen = sen + ' .'
    #             if sen not in ts:
    #                 ts.append(sen)
    #     triples = ts
    # make correfrence resolution for the triples extracted from the document.

    triples = cor_resolu_triple(triples, title)

    # triples.append(paragraph['text'])  # path retriever
    return triples

def encode_paras(paras,tokenizer,max_triple_length):
    '''
    :param paras: for the open-link paras, without triples. so we only encode the original text.
    :param tokenizer:
    :param max_triple_length: 512
    :return: the encoded id sequence for the paras.
    '''
    if type(paras)==str:# if the para is a long sentence.
        paras=paras.split('.')
        # tokenizedi = tokenizer.tokenize(paras)
        # tokenizedi = ['[CLS]'] + tokenizedi + ['[SEP]']
        # paras_ids = (tokenizer.convert_tokens_to_ids(tokenizedi))
        #
        # if len(paras_ids) > max_triple_length:
        #     paras_ids = paras_ids[:max_triple_length]
        # else:
        #     padding = [0] * (max_triple_length - len(paras_ids))
        #     paras_ids += padding
        # return [paras_ids]
    # elif type(paras)==list:# if the para is a set of sentences.
    encoded_output = []
    for i in paras:
        tokenizedi = tokenizer.tokenize(i)
        tokenizedi = ['[CLS]'] + tokenizedi + ['[SEP]']
        triple_ids = (tokenizer.convert_tokens_to_ids(tokenizedi))

        if len(triple_ids) > max_triple_length:
            triple_ids = triple_ids[:max_triple_length]
        else:
            padding = [0] * (max_triple_length - len(triple_ids))
            triple_ids += padding
        encoded_output.append(triple_ids)

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
    encoded_output.append(full_para_ids)

    return encoded_output

def convert_examples_to_features(examples,tokenizer,max_para_num = 500,max_seq_length=256,max_triples_size=40):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_q = tokenize_question(example.question, tokenizer, max_seq_length)

        q_triples = encode_triples(example.question_t, tokenizer, max_seq_length)
        question_embeds = tokenizer.convert_tokens_to_ids(tokens_q)
        if len(question_embeds) > max_seq_length:
            question_embeds = question_embeds[:max_seq_length]
        else:
            padding_qe = [0] * (max_seq_length - len(question_embeds))
            question_embeds += padding_qe

        q_triples.append(question_embeds)  #   the original question as a special triple+ extracted triples.
        num_qts = len(q_triples)
        num_pts = []

        qt_pt_input = []
        title2index = {}

        titles_list = list(example.context.keys())
        for p in titles_list:
            if len(qt_pt_input) == max_para_num:
                break

            if p in title2index:
                continue

            title2index[p] = len(title2index)
            example.title_order.append(p)
            title=p
            p = example.context[p]# p from key to value

            triples = fetchT_eval_para(p,title)
            p_triples = encode_triples(triples, tokenizer, max_seq_length)

            triples_special = encode_paras(p['text'], tokenizer, max_seq_length)  # -1 is the paragraph text.
            p_triples.extend(triples_special)

            num_pts_ = len(p_triples)
            num_pts.append(num_pts_)

            # the total triples for the current instance: the question triples + the para triples.
            triple_input_ids = q_triples + p_triples
            if len(triple_input_ids) < max_triples_size:  # it should be 36
                for lack in range(max_triples_size - len(triple_input_ids)):
                    triple_input_ids.append([0] * max_seq_length)
            triple_input_idsx = scipy.sparse.csc_matrix(np.array(triple_input_ids[:max_triples_size]))
            qt_pt_input.append(triple_input_idsx)


        num_paragraphs_no_links = len(qt_pt_input)

        assert len(qt_pt_input) <= max_para_num

        num_paragraphs = len(qt_pt_input)
        output_masks = [([1.0] * len(qt_pt_input) + [0.0] * (max_para_num - len(qt_pt_input) + 1)) for _ in
                        range(max_para_num + 2)]

        assert len(example.context) == num_paragraphs_no_links
        for i in range(len(output_masks[0])):
            if i >= num_paragraphs_no_links:
                output_masks[0][i] = 0.0

        for i in range(len(qt_pt_input)):
            output_masks[i + 1][i] = 0.0


        DUMMY_t = [0] * max_seq_length
        padding_triple = [DUMMY_t] * max_triples_size
        for lack in range(max_para_num-len(qt_pt_input)):
        	qt_pt_input.append(scipy.sparse.csc_matrix(np.array(padding_triple)))
        # num_qts = num_qts[0]*(max_para_num-len(num_qts))
        num_pts += [0] * (max_para_num - len(num_pts))

        features.append(
            InputFeatures(
                          q_ts_num=num_qts,
                          p_ts_num=num_pts,
                          qt_pts=qt_pt_input,
                          output_masks=output_masks,
                          num_paragraphs=num_paragraphs,
                          num_steps=-1,
                          ex_index=ex_index))
        #print('***finish process No.%d data.***'%ex_index)

    return features

class GraphTripleRetriever:
    def __init__(self,
                 args,
                 device):

        self.graph_retriever_config = GraphRetrieverConfig(example_limit=None,
                                                           task=None,
                                                           max_seq_length=args.max_seq_length,
                                                           max_select_num=args.max_select_num,
                                                           max_para_num=args.max_para_num,
                                                           tfidf_limit=None,
                                                           train_batchsize=4,

                                                           train_file_path=None,
                                                           use_redundant=None,
                                                           use_multiple_redundant=None,
                                                           max_redundant_num=None,

                                                           dev_file_path=None,
                                                           beam=args.beam_graph_retriever,
                                                           min_select_num=args.min_select_num,
                                                           no_links=args.no_links,
                                                           pruning_by_links=args.pruning_by_links,
                                                           expand_links=args.expand_links,
                                                           eval_chunk=args.eval_chunk,
                                                           tagme=args.tagme,
                                                           topk=args.topk,
                                                           db_save_path=None)

        print('initializing GraphRetriever...', flush=True)
        mox.file.copy_parallel("s3://obs-app-2020042019121301221/SEaaKM/W84184668/code/test/transformer_tokenizer/",
                               "cache/transformer_tokenizer/")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                       cache_dir=os.path.join("cache/", "transformer_tokenizer/"),
                                                       do_lower_case=args.do_lower_case, local_files_only=True)

        model_name = args.model_name
        mox.file.copy_parallel(
            "s3://obs-app-2020042019121301221/SEaaKM/W84184668/models/" + args.model_dir + "/" + model_name,
            "cache/" + model_name)
        graph_retriever_path = os.path.join("cache/", model_name)
        model_state_dict = torch.load(graph_retriever_path)
        mox.file.copy_parallel(
            "s3://obs-app-2020042019121301221/SEaaKM/W84184668/code/model_code_run/transformer_bert/",
            "cache/transformer_bert/")
        # state_dcit This option can be used if you want to create a model from a pretrained configuration but load your own weights.
        self.model = BertForGraphRetriever.from_pretrained(args.bert_model_graph_retriever,
                                                           cache_dir=os.path.join("cache/", "transformer_bert/"),
                                                           local_files_only=True,
                                                           state_dict=model_state_dict,
                                                           graph_retriever_config=self.graph_retriever_config)
        self.device = device
        self.model.to(self.device)
        self.model.eval()


        np_load_old = np.load
        # modify the default parameters of np.load
        np.load1 = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

        print('Done!', flush=True)

    def predict(self,
                tfidf_retrieval_output,
                retriever,
                args
                ):

        pred_output = []

        eval_examples = create_examples(tfidf_retrieval_output, self.graph_retriever_config, text_type='text',
                                        tripletype='all')

        TOTAL_NUM = len(eval_examples)
        eval_start_index = 0
        npy_idx = 0

        while eval_start_index < TOTAL_NUM:
            eval_end_index = min(eval_start_index + self.graph_retriever_config.eval_chunk - 1, TOTAL_NUM - 1)
            chunk_len = eval_end_index - eval_start_index + 1

            features = convert_examples_to_features(eval_examples[eval_start_index:eval_start_index + chunk_len],
                                                    args.max_seq_length, args.max_para_num, self.graph_retriever_config,
                                                    self.tokenizer, glove_dict=self.glove)
            npy_dirname = 'eval_' + str(args.human_weight) + 'eval_sample_' + str(npy_idx) + '.npy'

            np.save(os.path.join('/home/work/user-job-dir/model_code_run/', npy_dirname), features)
            mox.file.copy_parallel(os.path.join('/home/work/user-job-dir/model_code_run/', npy_dirname),
                                   "s3://obs-app-2020042019121301221/SEaaKM/W84184668/eval_data/npy_dir1/" + npy_dirname)
            print('******finished save the npy file.******')
            npy_idx = npy_idx + 1
            all_input_ids = torch.tensor([f.input_ids[0] for f in features], dtype=torch.long)
            all_input_masks = torch.tensor([f.input_masks for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_output_masks = torch.tensor([f.output_masks for f in features], dtype=torch.float)
            all_num_paragraphs = torch.tensor([f.num_paragraphs for f in features], dtype=torch.long)
            all_num_steps = torch.tensor([f.num_steps for f in features], dtype=torch.long)
            all_ex_indices = torch.tensor([f.ex_index for f in features], dtype=torch.long)

            # triple representation
            try:
                all_qt_pts = torch.tensor([f.qt_pts for f in features])
            except:

                tempdata = []
                for insdata in features:
                    if len(insdata.qt_pts) != 50:
                        one_qt_pts = insdata.qt_pts
                        one_qt_pts += (50 - len(insdata.qt_pts)) * [[[0] * 300] * 200]
                        tempdata.append(one_qt_pts)
                    else:
                        tempdata.append(insdata.qt_pts)
                all_qt_pts = torch.tensor(tempdata)

            all_qts_num = torch.tensor([f.q_ts_num for f in features], dtype=torch.long)
            try:
                all_pts_num = torch.tensor([f.p_ts_num[0] for f in features], dtype=torch.long)
            except:
                tempdata = []
                for insdata in features:
                    if len(insdata.p_ts_num[0]) != 50:
                        one_qt_pts = insdata.p_ts_num[0]
                        one_qt_pts += (50 - len(insdata.p_ts_num[0])) * [0]
                        tempdata.append(one_qt_pts)
                    else:
                        tempdata.append(insdata.p_ts_num[0])
                all_pts_num = torch.tensor(tempdata, dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_output_masks,
                                      all_num_paragraphs, all_num_steps, all_ex_indices, all_qt_pts, all_qts_num,
                                      all_pts_num)

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            logger.info('Examples from ' + str(eval_start_index) + ' to ' + str(eval_end_index))
            for input_ids, input_masks, segment_ids, output_masks, num_paragraphs, num_steps, ex_indices, all_qt_pts, all_qts_num, all_pts_num in tqdm(
                    eval_dataloader, desc="Evaluating"):

                batch_max_len = input_masks.sum(dim=2).max().item()
                batch_max_para_num = num_paragraphs.max().item()
                batch_max_steps = num_steps.max().item()

                input_ids = input_ids[:, :batch_max_para_num, :batch_max_len]
                input_masks = input_masks[:, :batch_max_para_num, :batch_max_len]
                segment_ids = segment_ids[:, :batch_max_para_num, :batch_max_len]
                output_masks = output_masks[:, :batch_max_para_num + 2, :batch_max_para_num + 1]
                output_masks[:, 1:, -1] = 1.0  # Ignore EOS in the first step
                all_qt_pts = all_qt_pts[:, :batch_max_para_num, :]
                all_qts_num = all_qts_num[:]
                all_pts_num = all_pts_num[:, :batch_max_para_num]

                input_ids = input_ids.to(self.device)
                input_masks = input_masks.to(self.device)
                segment_ids = segment_ids.to(self.device)
                all_qt_pts = all_qt_pts.to(self.device)
                all_qts_num = all_qts_num.to(self.device)
                all_pts_num = all_pts_num.to(self.device)

                output_masks = output_masks.to(self.device)

                examples = [eval_examples[eval_start_index + ex_indices[i].item()] for i in range(input_ids.size(0))]

                with torch.no_grad():
                    pred, prob, topk_pred, topk_prob = self.model.beam_search(args.human_weight, input_ids, segment_ids,
                                                                              input_masks, all_qt_pts, all_qts_num,
                                                                              all_pts_num, examples=examples,
                                                                              tokenizer=self.tokenizer,
                                                                              retriever=retriever,
                                                                              split_chunk=args.split_chunk)

                for i in range(len(pred)):
                    e = examples[i]

                    titles = [e.title_order[p] for p in pred[i]]
                    question = e.question

                    pred_output.append({})
                    pred_output[-1]['q_id'] = e.guid

                    pred_output[-1]['question'] = question

                    topk_titles = [[e.title_order[p] for p in topk_pred[i][j]] for j in range(len(topk_pred[i]))]
                    pred_output[-1]['topk_titles'] = topk_titles

                    topk_probs = []
                    pred_output[-1]['topk_probs'] = topk_probs

                    context = {}
                    context_from_tfidf = set()
                    context_from_hyperlink = set()
                    for ts in topk_titles:
                        for t in ts:
                            context[t] = e.all_paras[t]

                            if t in e.context:
                                context_from_tfidf.add(t)
                            else:
                                context_from_hyperlink.add(t)

                    pred_output[-1]['context'] = context
                    pred_output[-1]['context_from_tfidf'] = list(context_from_tfidf)
                    pred_output[-1]['context_from_hyperlink'] = list(context_from_hyperlink)

            eval_start_index = eval_end_index + 1
            del features
            del all_input_ids
            del all_input_masks
            del all_segment_ids
            del all_output_masks
            del all_num_paragraphs
            del all_num_steps
            del all_ex_indices
            del eval_data

        return pred_output

def process_examples():
    # dirname='data\eval_data_merged_redun2'
    # files=getfiles(dirname)
    # features=create_examples(files)
    with open('data/eval_feature.pkl', 'rb')as f:
        features= pickle.load(f)
    num_files=30
    chunk=int(len(features)/num_files)
    for i in range(num_files):
        start=chunk*i
        end=start+chunk
        if len(features)-end<chunk:
            end=len(features)
        print('store %d to %d data to file %s.' % (start, end, str(i)))
        with open('eval_feature_'+str(i)+'.pkl','wb')as f:
            pickle.dump(features[start:end],f)

    '''
    store 0 to 1851 data to file 0.
store 1851 to 3702 data to file 1.
store 3702 to 5553 data to file 2.
store 5553 to 7405 data to file 3.

    '''
from transformers import BertTokenizer
def convert_example_2_feature():
    mox.file.copy_parallel(
        "s3://obs-app-2019071708472901661/w84184668/eval_data/pkl_examples_dir/pkl_files_eval/",
        "/cache")
    location_dir = mox.file.list_directory("/cache", recursive=True)
    filelists=[]
    for file in location_dir:
        if '.pkl' in file:
            filelists.append(os.path.join('/cache',file))
    # already=['eval_feature_0.pkl','eval_feature_10.pkl','eval_feature_1.pkl','eval_feature_11.pkl','eval_feature_12.pkl','eval_feature_13.pkl',
    #          'eval_feature_14.pkl','eval_feature_15.pkl','eval_feature_16.pkl','eval_feature_17.pkl','eval_feature_18.pkl','eval_feature_19.pkl',
    #          'eval_feature_2.pkl','eval_feature_20.pkl','eval_feature_21.pkl','eval_feature_22.pkl','eval_feature_23.pkl','eval_feature_24.pkl',
    #          'eval_feature_25.pkl','eval_feature_26.pkl','eval_feature_27.pkl','eval_feature_28.pkl']
    #
    need_modify=['eval_feature_22.pkl']#28 ,'eval_feature_21.pkl','eval_feature_27.pkl' 23

    print('load %d files.'%len(filelists))
    #filelists=["eval_feature_0.pkl","eval_feature_1.pkl","eval_feature_2.pkl","eval_feature_3.pkl"]

    mox.file.copy_parallel(
        "s3://obs-app-2019071708472901661/w84184668/code/transformers_tokenizer/",
        "cache/transformers_tokenizer/")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=os.path.join("cache/","transformers_tokenizer/"), local_files_only=True)

    for file in tqdm(filelists):
        file_pure=file.split('/')[-1]
        if file_pure in need_modify:
            data = read(file)
            print('***start process %d datas of file %s.***'%(len(data),file))
            features=convert_examples_to_features(data,tokenizer)
            output_filename = file.replace('.pkl','.npy')
            output_filename='/home/work/user-job-dir/code/'+output_filename.split('/')[-1]

            np.save(output_filename, features)
            print('finish process file %s.'%output_filename)
            mox.file.copy_parallel(
                output_filename,
                "s3://obs-app-2019071708472901661/w84184668/eval_data/npy_features_dir/"+output_filename.split('/')[-1])

from tqdm import tqdm
import argparse
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',default=0,type=int)
    parser.add_argument('--end',default=30,type=int)
    parser.add_argument('--input',default='eval_data/eval_feature_1.pkl',type=str)

    args=parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                              cache_dir="model_dir/transformer_tokenizer",
                                              local_files_only=True)
    filenames=getfiles('eval_data')
    if args.end > len(filenames):
        end=len(filenames)
    else:
        end=args.end
    if os.path.exists('eval_data_npy'):
        print('output dir exists!')
    else:
        os.mkdir('eval_data_npy')
    filenames=['eval_data/eval_feature_22.pkl','eval_data/eval_feature_23.pkl','eval_data/eval_feature_14.pkl','eval_data/eval_feature_17.pkl','eval_data/eval_feature_19.pkl']
    filenames=[args.input]
    print(filenames)
    for file in filenames[:]:
        print('start process file：%s.'%file)
        data=read(file)
        features = convert_examples_to_features(data, tokenizer,500,256,40)
        output=file.replace('.pkl','.npy')
        output=output.replace('eval_data','eval_data_npy')
        np.save(output,features)
        assert len(features)==len(data)
        print('finish dump the %d data to file %s.'%(len(features),output))





if __name__ == '__main__':
    main()
    '''392 questions do not have extracted triples.
    7495 in total.
    '''
    # dirname='pkl_files_eval'
    # filenames=getfiles(dirname)
    # total_com=0
    # total_bri=0
    # eval_ground=read('data/ground_type_id.json')
    # # for key in eval_ground:
    # #     if eval_ground[key]['type']=='comparison':
    # #         total_com+=1
    # #     elif eval_ground[key]['type']=='bridge':
    # #         total_bri+=1
    # # print(total_com)
    # # print(total_bri)
    # # print('*'*20)
    # comparasion_yes=0
    # comparasion_no=0
    # bridge_yes=0
    # bridge_no=0
    # comparasion_triple_dict=dict()
    # bridge_triple_dict=dict()
    # for file in filenames:
    #     data=read(file)
    #     for idx in range(len(data)):
    #         ground=eval_ground[str(data[idx].guid)]
    #         type=ground['type']
    #         if type=='comparison':
    #             ground_title=list(set(ground['ground_p']))
    #             click=0
    #             ground_title = list(set(ground['ground_p']))
    #             for title in ground_title:
    #                 if title not in data[idx].context:
    #                     continue
    #                 num=len(data[idx].context[title]['merged_triples'])
    #                 if num in comparasion_triple_dict:
    #                     comparasion_triple_dict[num]=comparasion_triple_dict[num]+1
    #                 else:
    #                     comparasion_triple_dict[num]=1
    #                 # if len(data[idx].context[title]['merged_triples'])==0:
    #                 #     comparasion_no+=1
    #                 # else:
    #                 #     comparasion_yes+=1
    #         elif type=='bridge':
    #             ground_title = list(set(ground['ground_p']))
    #             click = 0
    #             for title in ground_title:
    #                 if title not in data[idx].context:
    #                     continue
    #                 num = len(data[idx].context[title]['merged_triples'])
    #                 if num in bridge_triple_dict:
    #                     bridge_triple_dict[num]=bridge_triple_dict[num]+1
    #                 else:
    #                     bridge_triple_dict[num]=1
    #
    #                 # if len(data[idx].context[title]['merged_triples'])==0:
    #                 #     bridge_no += 1
    #                 #     # click = 1
    #                 #     # break
    #                 # else:
    #                 #     bridge_yes += 1
    #
    # # print(comparasion_yes)
    # # print(comparasion_no)
    # # print(bridge_yes)
    # # print(bridge_no)
    # print(collections.OrderedDict(sorted(bridge_triple_dict.items())))
    # print(collections.OrderedDict(sorted(comparasion_triple_dict.items())))
    #




    #convert_example_2_feature()
    # filename='pkl_files_eval/eval_feature_0.pkl'
    # data=read(filename)
    # ground=read('data/ground_type_id.json')
    # groundp=ground[data[0].guid]
    # context_keys=list(data[0].context.keys())
    # for idx,i in enumerate(context_keys):
    #     if i in groundp['ground_p']:
    #         print(idx)
    #
    # print(groundp)
    #process_examples()
    #main()
    # np_load_old = np.load
    # # modify the default parameters of np.load
    # np.load1 = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    #
    # data=np.load1('eval_feature_0.npy')
    # print(data[0].qt_pts)

