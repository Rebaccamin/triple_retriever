# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import json

import scipy
from scipy.constants import pi

print("sciPy - pi = %.16f" % pi)

import numpy as np

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers import PYTORCH_PRETRAINED_BERT_CACHE
from pathlib import Path
PRETRAINED_BERT_CACHE = Path("/export/data/chengminu/.transformers")
import gc
import pickle
import time
from model import BertForGraphRetriever
from utils_triple import DataProcessor
from utils_triple import  convert_examples_to_features, save_feature2dir
from utils_triple import save, load
from utils_triple import GraphRetrieverConfig
from utils_triple import warmup_linear


import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def getfiles(dirname):
    filenames = []
    for dir, _, files in os.walk(dirname):
        for i in files:
                filenames.append(os.path.join(dir, i))
    return filenames

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--task',
                        type=str,
                        default=None,
                        required=True,
                        help="Task code in {hotpot_open, hotpot_distractor, squad, nq}")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=5,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam. (def: 5e-5)")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # specific parameters for triple input

    parser.add_argument('--max_triples_size', default=64, type=int)

    # RNN graph retriever-specific parameters
    parser.add_argument("--example_limit",
                        default=None,
                        type=int)

    parser.add_argument("--max_para_num",
                        default=50,
                        type=int)
    parser.add_argument("--neg_chunk",
                        default=1,
                        type=int,
                        help="The chunk size of negative examples during training (to reduce GPU memory consumption with negative sampling)")
    parser.add_argument("--eval_chunk",
                        default=100000,
                        type=int,
                        help="The chunk size of evaluation examples (to reduce RAM consumption during evaluation)")
    parser.add_argument("--split_chunk",
                        default=300,
                        type=int,
                        help="The chunk size of BERT encoding during inference (to reduce GPU memory consumption)")

    parser.add_argument('--train_file_path',
                        type=str,
                        default=None,
                        help="File path to the training data")
    parser.add_argument('--dev_file_path',
                        type=str,
                        default=None,
                        help="File path to the eval data")

    parser.add_argument('--beam',
                        type=int,
                        default=1,
                        help="Beam size")
    parser.add_argument('--min_select_num',
                        type=int,
                        default=1,
                        help="Minimum number of selected paragraphs")
    parser.add_argument('--max_select_num',
                        type=int,
                        default=4,
                        help="Maximum number of selected paragraphs")
    parser.add_argument("--use_redundant",
                        action='store_true',
                        help="Whether to use simulated seqs (only for training)")
    parser.add_argument("--use_multiple_redundant",
                        action='store_true',
                        help="Whether to use multiple simulated seqs (only for training)")
    parser.add_argument('--max_redundant_num',
                        type=int,
                        default=100000,
                        help="Whether to limit the number of the initial TF-IDF pool (only for open-domain eval)")
    parser.add_argument("--no_links",
                        action='store_true',
                        help="Whether to omit any links (or in other words, only use TF-IDF-based paragraphs)")
    parser.add_argument("--pruning_by_links",
                        action='store_true',
                        help="Whether to do pruning by links (and top 1)")
    parser.add_argument("--expand_links",
                        action='store_true',
                        help="Whether to expand links with paragraphs in the same article (for NQ)")
    parser.add_argument('--tfidf_limit',
                        type=int,
                        default=40,
                        help="Whether to limit the number of the initial TF-IDF pool (only for open-domain eval)")

    parser.add_argument("--pred_file", default=None, type=str,
                        help="File name to write paragraph selection results")
    parser.add_argument('--topk',
                        type=int,
                        default=2,
                        help="Whether to use how many paragraphs from the previous steps")

    parser.add_argument("--model_suffix", default=None, type=str,
                        help="Suffix to load a model file ('pytorch_model_' + suffix +'.bin')")

    parser.add_argument("--db_save_path", default=None, type=str,
                        help="File path to DB")
    parser.add_argument("--cuda_device", default=7, type=int)
    parser.add_argument("--converted_feature_dir", type=str)

    parser.add_argument("--load_data", default=10, type=int)
    parser.add_argument("--glovefile", default="glove.json", type=str)
    parser.add_argument('--file_start_idx',default=2,type=int)

    args = parser.parse_args()

    cpu = torch.device('cpu')

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    n_gpu = 1# torch.cuda.device_count()
    #print('****Num of gpus: %d.****'%n_gpu)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    if args.train_file_path is not None:
        do_train = True


    elif args.dev_file_path is not None:
        do_train = False

    else:
        raise ValueError('One of train_file_path: {} or dev_file_path: {} must be non-None'.format(args.train_file_path,
                                                                                                   args.dev_file_path))

    processor = DataProcessor()

    # Configurations of the graph retriever
    graph_retriever_config = GraphRetrieverConfig(example_limit=args.example_limit,
                                                  task=args.task,
                                                  max_seq_length=args.max_seq_length,
                                                  max_select_num=args.max_select_num,
                                                  max_para_num=args.max_para_num,
                                                  tfidf_limit=args.tfidf_limit,
                                                  train_batchsize=args.train_batch_size,

                                                  train_file_path=args.train_file_path,
                                                  use_redundant=args.use_redundant,
                                                  use_multiple_redundant=args.use_multiple_redundant,
                                                  max_redundant_num=args.max_redundant_num,

                                                  dev_file_path=args.dev_file_path,
                                                  beam=args.beam,
                                                  min_select_num=args.min_select_num,
                                                  no_links=args.no_links,
                                                  pruning_by_links=args.pruning_by_links,
                                                  expand_links=args.expand_links,
                                                  eval_chunk=args.eval_chunk,
                                                  tagme=False,
                                                  topk=args.topk,
                                                  db_save_path=args.db_save_path)

    logger.info(graph_retriever_config)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, cache_dir=PRETRAINED_BERT_CACHE, do_lower_case=args.do_lower_case)

    ##############################
    # Training                   #
    ##############################

    if do_train:

        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            trained_model_files = []
            for dir, _, files in os.walk(args.output_dir):
                for file in files:
                    if '.bin' in file:
                        trained_model_files.append(os.path.join(dir, file))
            print('load trained model %s.' % trained_model_files[-1])
            model_state_dict = torch.load(trained_model_files[-1])
            model = BertForGraphRetriever.from_pretrained(args.bert_model,  state_dict=model_state_dict,
                                          graph_retriever_config=graph_retriever_config)
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            model = BertForGraphRetriever.from_pretrained(args.bert_model, cache_dir=PRETRAINED_BERT_CACHE,
                                                          graph_retriever_config=graph_retriever_config)

        model.to(device)



        if os.path.exists(args.converted_feature_dir) and os.path.isdir(args.converted_feature_dir):
            np_load_old = np.load
            # modify the default parameters of np.load
            np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
            train_features = []

            converted_feature_files = getfiles(args.converted_feature_dir)
            for idx, location in enumerate(converted_feature_files):
                    train_features_numpy = np.load(location)
                    train_features += train_features_numpy.tolist()
                # else:
                #     break
        else:
            filenames= getfiles(graph_retriever_config.train_file_path)
            for fileidx,file in enumerate(filenames):
                print('start process file.')
                if fileidx>-1:
                    #train_examples = processor.get_train_examples(graph_retriever_config,file)
                    with open(file,'rb')as f:
                        train_examples = pickle.load(f)
                    train_features = convert_examples_to_features(
                        train_examples, args.max_triples_size, args.max_seq_length, args.max_para_num,
                        graph_retriever_config, tokenizer, train=True)
                    del train_examples
                    gc.collect()
                    save_feature2dir(train_features, args.converted_feature_dir,fileidx)
                    del train_features
                    gc.collect()


        print('****start load model and train (%d data)****' % len(train_features))

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        POSITIVE = 1.0
        NEGATIVE = 0.0

        # len(train_examples) and len(train_features) are different, by the redundant gold.
        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps

        optimizer = AdamW(optimizer_grouped_parameters[0]['params'],
                          lr=args.learning_rate,
                          correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(t_total*args.warmup_proportion),num_training_steps=t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        model.train()
        torch.backends.cudnn.benchmark = True
        epc = 0
        for _ in range(int(args.num_train_epochs)):
            logger.info('Epoch ' + str(epc + 1))

            TOTAL_NUM = len(train_features)
            train_start_index = 0
            CHUNK_NUM = 10
            train_chunk = TOTAL_NUM // CHUNK_NUM
            chunk_index = 0

            #random.shuffle(train_features)


            while train_start_index < TOTAL_NUM:
                train_end_index = min(train_start_index + train_chunk - 1, TOTAL_NUM - 1)
                chunk_len = train_end_index - train_start_index + 1

                train_features_ = train_features[train_start_index:train_start_index + chunk_len]
                all_output_masks = torch.tensor([f.output_masks for f in train_features_], dtype=torch.float)
                all_num_paragraphs = torch.tensor([f.num_paragraphs for f in train_features_], dtype=torch.long)
                all_num_steps = torch.tensor([f.num_steps for f in train_features_], dtype=torch.long)

                try:
                    all_qt_pts = []
                    for f in train_features_:
                        qt_pts = f.qt_pts
                        new_qt_pts = []
                        for i in qt_pts:
                            new_qt_pts.append(i.todense())
                        all_qt_pts.append(np.array(new_qt_pts))
                    all_qt_pts = np.array(all_qt_pts)
                    all_qt_pts = torch.tensor(all_qt_pts, dtype=torch.long)
                except:
                    print(train_start_index)
                    return

                all_qts_num = torch.tensor([f.q_ts_num for f in train_features_], dtype=torch.long)
                all_pts_num = torch.tensor([f.p_ts_num[0] for f in train_features_], dtype=torch.long)


                train_data = TensorDataset(all_output_masks,
                                           all_num_paragraphs, all_num_steps, all_qt_pts, all_qts_num, all_pts_num)

                train_sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

                tr_loss = 0

                logger.info('Examples from ' + str(train_start_index) + ' to ' + str(train_end_index))
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                        batch_max_len = (batch[4]+batch[5]).max().item() # 一个batch 内所有para（'q_t+p_t'）最大的数据量。

                        num_paragraphs = batch[1]
                        batch_max_para_num = num_paragraphs.max().item()  # 一个batch内 最多的paras 数量。

                        num_steps = batch[2]


                        batch = tuple(t.to(device) for t in batch)
                        output_masks, _, _, qt_pts, qt_num, pt_num = batch
                        B = qt_pts.size(0)

                        qt_pts = qt_pts[:, :batch_max_para_num, :batch_max_len]
                        qt_num = qt_num[:]
                        pt_num = pt_num[:, :batch_max_para_num]

                        target = torch.FloatTensor(qt_pts.size(0),qt_pts.size(1)).fill_(NEGATIVE)
                        for i in range(B):
                            for j in range(num_steps[i].item() - 1):
                                target[i, j].fill_(POSITIVE)
                        target = target.to(device)

                        train_start = 0
                        while train_start < batch_max_para_num:  # short_gold+context :

                            target_ = target[:, train_start:train_start + 1]
                            input_qt_pt_ = qt_pts[:, train_start:train_start + 1, :]
                            input_pt_nums = pt_num[:, train_start:train_start + 1]
                            input_qt_nums = qt_num[:]


                            loss = model(target_,input_qt_pt_, input_pt_nums, input_qt_nums)



                            if n_gpu > 1:
                                loss = loss.mean()  # mean() to average on multi-gpu.
                            if args.gradient_accumulation_steps > 1:
                                loss = loss / args.gradient_accumulation_steps

                            loss.backward()
                            tr_loss += loss.item()
                            train_start = train_start + 1

                            max_grad_norm = 1.0

                            # modify learning rate with special warm up BERT uses
                            lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                            torch.nn.utils.clip_grad_norm(optimizer_grouped_parameters[0]['params'],max_grad_norm)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            global_step += 1
                            #torch.cuda.empty_cache()

                            del target_
                            del input_qt_pt_
                            del input_pt_nums
                            del input_qt_nums
                            del loss

                        del batch
                        del output_masks
                        del qt_pts
                        del qt_num
                        del pt_num
                        del target


                chunk_index += 1
                train_start_index = train_end_index + 1

                # Save the model at the half of the epoch
                #if (chunk_index == CHUNK_NUM // 2 or save_retry):
                status = save(model, args.output_dir, str(chunk_index/100 + epc))
                #save_retry = (not status)

                del all_output_masks
                del all_num_paragraphs
                del all_num_steps
                del all_qt_pts
                del all_pts_num
                del all_qts_num
                del train_data
                torch.cuda.empty_cache()

            # Save the model at the end of the epoch
            save(model, args.output_dir, str(epc + 1))

            epc += 1

    if do_train:
        return


if __name__ == "__main__":
    main()
