from __future__ import division
from __future__ import print_function

import os

os.system("pip install tqdm")
os.system("pip install transformers")
os.system("pip install torch")
os.system("pip install scipy")
os.system("pip install tqdm")
os.system("pip install sqlitedict")
os.system("pip install nltk")
os.system("pip install tensorboard")
# os.system("pip install nltk")
import logging
import argparse
import random
from tqdm import tqdm, trange
import json
from torch.utils.tensorboard import SummaryWriter

import scipy
from scipy.constants import pi
import moxing as mox

print("sciPy - pi = %.16f" % pi)

import numpy as np

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.distributed as dist
from transformers import BertTokenizer

from transformers import BertPreTrainedModel, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import PYTORCH_PRETRAINED_BERT_CACHE
from pathlib import Path

PRETRAINED_BERT_CACHE = Path("/export/data/chengminu/.transformers")
import gc
import pickle
import time
from torch.utils.data.distributed import DistributedSampler
from model import BertForGraphRetriever
from utils_triple import DataProcessor
from utils_triple import convert_examples_to_features, save_feature2dir
from utils_triple import save, load
from utils_triple import GraphRetrieverConfig
from utils_triple import warmup_linear
from para_encoder import Para_encoder
import nltk
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from config import train_args

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


np_load_old = np.load
np.load1 = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def load_data(filenames, start, end):
    train_features = []
    if len(filenames) - end < 10:
        end = len(filenames)
        print('load the last %d files. (%d,%d).' % (end - start, start, end))

    for i in range(start, end):
        logger.info("load data from " + filenames[i])
        train_features_numpy = np.load1(filenames[i])
        train_features += train_features_numpy.tolist()
    return train_features


def write_file(data, file):
    outfile = file.replace('.pkl', '.npy')
    outfile = outfile.split('/')[-1]
    np.save("/home/work/user-job-dir/code/" + outfile, data)
    mox.file.copy_parallel("/home/work/user-job-dir/code/" + outfile,
                           "s3://obs-app-2019071708472901661/w84184668/train_data/feature_data_npy_dir_256_40/" + outfile)

    return


def writer_log(logfile):
    logfile = logfile + '.log'
    tb_writer = SummaryWriter('/home/work/user-job-dir/code/' + logfile)

    return tb_writer


def setup(args, global_rank):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.local_rank != -1:
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.init_method,
                                world_size=args.world_size,
                                rank=global_rank)


def load_model(args, graph_retriever_config):
    # np_load_old = np.load
    # # modify the default parameters of np.load
    # np.load_mm = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    mox.file.copy_parallel(
        "s3://obs-app-2019071708472901661/w84184668/code/bert_model/",
        "cache/transformer_bert/")

    if args.para_encoder:
        mox.file.copy_parallel(
            "s3://obs-app-2019071708472901661/w84184668/para_encoder_model/para_encoder_model_2.bin",
            "cache/para_encoder_model_2.bin")

        model_state_dict = torch.load('cache/' + 'para_encoder_model_2.bin')
        Para_encoder_model = Para_encoder.from_pretrained('bert-base-uncased',
                                                          cache_dir=os.path.join("cache/", "transformer_bert/"),
                                                          state_dict=model_state_dict,
                                                          local_files_only=True)

        if args.continue_train:
            print('continue train from %s.' % args.trained_model_dict)
            mox.file.copy_parallel(
                "s3://obs-app-2019071708472901661/w84184668/hop1_model/" + args.output_dir + "/" + args.trained_model_dict,
                'cache/' + args.trained_model_dict)
            state_dict_name = 'cache/' + args.trained_model_dict
            state_dict_model = torch.load(state_dict_name)

            model = BertForGraphRetriever.from_pretrained('bert-base-uncased',
                                                          cache_dir=os.path.join("cache/", "transformer_bert/"),
                                                          local_files_only=True,
                                                          state_dict=state_dict_model,

                                                          graph_retriever_config=graph_retriever_config)
        else:  # os.path.join("cache/", "transformer_bert/")
            mox.file.copy_parallel(
                "s3://obs-app-2019071708472901661/w84184668/code/bert_model/",
                "cache/transformer_bert/")
            try:
                model = BertForGraphRetriever.from_pretrained('bert-base-uncased',
                                                              cache_dir="cache/transformer_bert/",
                                                              local_files_only=True,
                                                              para_encoder_model=Para_encoder_model,
                                                              graph_retriever_config=graph_retriever_config)
            except:
                try:
                    model = BertForGraphRetriever.from_pretrained('bert-base-uncased',
                                                                  cache_dir="/home/work/user-job-dir/code/bert_model/",
                                                                  local_files_only=True,
                                                                  para_encoder_model=Para_encoder_model,
                                                                  graph_retriever_config=graph_retriever_config)
                except:
                    print('********************************************************')
                    return None
        # model = BertForGraphRetriever.from_pretrained('bert-base-uncased',
        #                                               cache_dir=os.path.join("cache/", "transformer_bert/"),
        #                                               local_files_only=True, para_encoder_model=Para_encoder_model,
        #                                               graph_retriever_config=graph_retriever_config)
    else:
        if args.continue_train:
            print('start continue train after %s.'%args.trained_model_dict)
            mox.file.copy_parallel(
                "s3://obs-app-2019071708472901661/w84184668/hop1_model/" + args.output_dir + "/" + args.trained_model_dict,
                'cache/' + args.trained_model_dict)
            state_dict_name = 'cache/' + args.trained_model_dict
            state_dict_model = torch.load(state_dict_name)

            model = BertForGraphRetriever.from_pretrained('bert-base-uncased',
                                                          cache_dir=os.path.join("cache/", "transformer_bert/"),
                                                          local_files_only=True,
                                                          state_dict=state_dict_model,
                                                          para_encoder_model=None,
                                                          graph_retriever_config=graph_retriever_config)
        else:  # os.path.join("cache/", "transformer_bert/")
            mox.file.copy_parallel(
                "s3://obs-app-2019071708472901661/w84184668/code/bert_model/",
                "cache/transformer_bert/")
            try:
                model = BertForGraphRetriever.from_pretrained('bert-base-uncased',
                                                              cache_dir="cache/transformer_bert/",
                                                              local_files_only=True,
                                                              para_encoder_model=None,
                                                              graph_retriever_config=graph_retriever_config)
            except:
                try:
                    model = BertForGraphRetriever.from_pretrained('bert-base-uncased',
                                                                  cache_dir="/home/work/user-job-dir/code/bert_model/",
                                                                  local_files_only=True,
                                                                  para_encoder_model=None,
                                                                  graph_retriever_config=graph_retriever_config)
                except:
                    print('********************************************************')
                    return None

    return model


def train(local_rank, args, ngpus_per_node,train_features):
    tb_logger = writer_log(args.logfile)
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

    args.local_rank = local_rank
    global_rank = args.rank * (ngpus_per_node) + local_rank
    if args.local_rank != -1:
        args.global_rank = args.rank * ngpus_per_node + args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    setup(args, global_rank)

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
                                                  tagme=None,
                                                  topk=args.topk,
                                                  db_save_path=None)
    model = load_model(args, graph_retriever_config)
    model.to(device)

    if dist.is_initialized():
        # if dist.get_rank() == 0:
        #     print(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=args.local_rank,
            find_unused_parameters=True
        )
    else:
        print(model.__class__.__name__)

    global_step = 0
    num_len_train_features = len(train_features)  # 115078

    # len(train_examples) and len(train_features) are different, by the redundant gold.
    num_train_steps = int(
        num_len_train_features / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_len_train_features)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    model.train()
    torch.backends.cudnn.benchmark = True
    epc = 0
    first = 0
    file_idx=0
    for epoch_num in range(int(args.num_train_epochs)):
            if epoch_num==0 and args.continue_train:
                continue
            logger.info('Epoch ' + str(epoch_num + 1))

            CHUNK_NUM = 20#10
            chunk_index = 0

            train_start_index = 0
            TOTAL_NUM = len(train_features)
            current_train_chunk = TOTAL_NUM // CHUNK_NUM

            while train_start_index < TOTAL_NUM:
                train_end_index = min(train_start_index + current_train_chunk - 1, TOTAL_NUM - 1)
                chunk_len = train_end_index - train_start_index + 1

                train_features_ = train_features[train_start_index:train_start_index + chunk_len]
                all_output = torch.tensor([f.output_masks for f in train_features_], dtype=torch.float)
                all_num_paragraphs = torch.tensor([f.num_paragraphs for f in train_features_], dtype=torch.long)
                all_para_contents = torch.tensor([f.para_contents for f in train_features_],
                                                 dtype=torch.long)  # 378 dim
                all_num_ground_triples = torch.tensor([f.p_gts_num[0] for f in train_features_], dtype=torch.long)

                # load the sparse triple matrix for each instance(question+a para)
                all_qt_pts = []
                for f in train_features_:
                    qt_pts = f.qt_pts[0]
                    new_qt_pts = []
                    for i in qt_pts:
                        new_qt_pts.append(i.todense())
                    all_qt_pts.append(new_qt_pts)
                all_qt_pts = np.array(all_qt_pts)
                all_qt_pts = torch.tensor(all_qt_pts, dtype=torch.long)

                all_qts_num = torch.tensor([f.q_ts_num for f in train_features_], dtype=torch.long)
                all_pts_num = torch.tensor([f.p_ts_num[0] for f in train_features_], dtype=torch.long)

                train_data = TensorDataset(all_output, all_num_paragraphs, all_para_contents, all_qt_pts, all_qts_num,
                                           all_pts_num, all_num_ground_triples)


                train_sampler = DistributedSampler(train_data)
                train_dataloader = DataLoader(train_data,
                                              sampler=train_sampler,
                                              batch_size=args.train_batch_size)

                tr_loss = 0
                print('Examples from ' + str(train_start_index) + ' to ' + str(train_end_index))
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch_max_len = (batch[4] + batch[5] + batch[
                        6]).max().item()
                    num_paragraphs = batch[1]
                    batch_max_para_num = num_paragraphs.max().item()  # 一个batch内 最多的paras 数量。

                    batch = tuple(t.to(device) for t in batch)
                    output_masks, _, para_contents, qt_pts, qt_num, pt_num, gt_num = batch

                    qt_pts = qt_pts[:, :batch_max_para_num, :batch_max_len]
                    para_content = para_contents[:, :batch_max_para_num]
                    gt_num = gt_num[:, :batch_max_para_num]
                    qt_num = qt_num[:]
                    pt_num = pt_num[:, :batch_max_para_num]

                    target = output_masks[:, :batch_max_para_num]

                    if args.negative_num != -1:
                        candidates = np.arange(2, batch_max_para_num).tolist()
                        candidates = random.sample(candidates, args.negative_num)
                        train_indexs = [0, 1] + candidates  # assert ground must covered.
                        train_indexs = list(set(train_indexs))
                    else:
                        train_indexs = np.arange(0, batch_max_para_num).tolist()

                    for train_start in train_indexs:  # short_gold+context:
                        target_ = target[:, train_start:train_start + 1]  # one instance
                        input_qt_pt_ = qt_pts[:, train_start:train_start + 1, :]
                        input_pt_nums = pt_num[:, train_start:train_start + 1]
                        input_qt_nums = qt_num[:]
                        input_gt_nums = gt_num[:, train_start:train_start + 1]
                        input_content = para_content[:, train_start:train_start + 1]

                        loss = model(target_, input_qt_pt_, input_content, input_pt_nums, input_qt_nums, input_gt_nums,
                                     args.max_triple_size,
                                     args.merge_way, args.model_type,args.metrics)

                        # if n_gpu > 1:
                        #     loss = loss.mean()  # mean() to average on multi-gpu.
                        # if args.gradient_accumulation_steps > 1:
                        #     loss = loss / args.gradient_accumulation_steps

                        loss.backward()
                        tr_loss += loss.item()
                        max_grad_norm = 1.0

                        # modify learning rate with special warm up BERT uses
                        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        torch.nn.utils.clip_grad_norm(optimizer_grouped_parameters[0]['params'], max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        tb_logger.add_scalar('batch_train_loss',
                                             loss.item(), global_step)
                        # print("current loss: %f." % loss.item())

                        global_step += 1

                        torch.cuda.empty_cache()

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
                    #
                    # print('finish train on no%d batch.' % (step))

                chunk_index += 1
                train_start_index = train_end_index + 1

                del all_num_paragraphs
                del all_qt_pts
                gc.collect()
                del all_pts_num
                del all_qts_num
                del train_data
                gc.collect()

                file_idx+=1
                save(model, "/home/work/user-job-dir/code/", str(round(epoch_num + file_idx / 100, 2)))
                mox.file.copy_parallel(
                    "/home/work/user-job-dir/code/pytorch_model_" + str(round(epoch_num + file_idx / 100, 2)) + ".bin",
                    "s3://obs-app-2019071708472901661/w84184668/hop1_model/" + args.output_dir + "/pytorch_model_" + str(
                        round(epoch_num + file_idx / 100, 2)) + ".bin")
                mox.file.copy_parallel("/home/work/user-job-dir/code/" + args.logfile + '.log',
                                   "s3://obs-app-2019071708472901661/w84184668/code/" + args.logfile + '.log')


            # Save the model at the end of the epoch
            save(model, "/home/work/user-job-dir/code/", str(epoch_num + 1))
            mox.file.copy_parallel(
                "/home/work/user-job-dir/code/pytorch_model_" + str(epoch_num + 1) + ".bin",
                "s3://obs-app-2019071708472901661/w84184668/hop1_model/" + args.output_dir + "/pytorch_model_" + str(
                    epoch_num + 1) + ".bin")
            mox.file.copy_parallel("/home/work/user-job-dir/code/" + args.logfile + '.log',
                               "s3://obs-app-2019071708472901661/w84184668/code/" + args.logfile + '.log')


def main():
    args = train_args()
    ngpus_per_node = torch.cuda.device_count()
    train_features=[]
    for fileidx in range(0, 92):
        filename = 'train_file_' + str(fileidx) + '.npy'
        mox.file.copy_parallel(
                "s3://obs-app-2019071708472901661/w84184668/train_data/hop1_npy_data/" + filename,
                "/cache/" + filename)
        filename = os.path.join('/cache', filename)
        try:
            train_features_ = np.load_mm(filename)
        except:
                np_load_old = np.load
                np.load_now = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
                train_features_ = np.load_now(filename)
        print('load %d data from file: %s.'%(len(train_features_),filename))
        train_features += train_features_.tolist()
    print('Start train on %d data......'%len(train_features))
    mp.spawn(train, nprocs=ngpus_per_node, args=(args, ngpus_per_node,train_features))


def main_old():
    args = train_args()
    tb_logger = writer_log(args.logfile)

    args.ngpus_per_node = torch.cuda.device_count()
    mp.spawn(train, nprocs=args.ngpus_per_node, args=(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Configurations of the triple retriever
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
                                                  tagme=None,
                                                  topk=args.topk,
                                                  db_save_path=None)

    logger.info(graph_retriever_config)

    ##############################
    # Training                   #
    ##############################
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load_mm = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    mox.file.copy_parallel(
        "s3://obs-app-2019071708472901661/w84184668/code/bert_model/",
        "cache/transformer_bert/")

    if args.para_encoder:
        mox.file.copy_parallel(
            "s3://obs-app-2019071708472901661/w84184668/para_encoder_model/",
            "cache/para_encoder_model/")

        model_state_dict = torch.load('cache/para_encoder_model/' + 'para_encoder_model_2.bin')
        Para_encoder_model = Para_encoder.from_pretrained('bert-base-uncased',
                                                          cache_dir=os.path.join("cache/", "transformer_bert/"),
                                                          state_dict=model_state_dict,
                                                          local_files_only=True)
        model = BertForGraphRetriever.from_pretrained('bert-base-uncased',
                                                      cache_dir=os.path.join("cache/", "transformer_bert/"),
                                                      local_files_only=True, para_encoder_model=Para_encoder_model,
                                                      graph_retriever_config=graph_retriever_config)
    else:
        if args.continue_train:
            mox.file.copy_parallel(
                "s3://obs-app-2019071708472901661/w84184668/hop1_model/" + args.output_dir + "/" + args.trained_model_dict,
                'cache/' + args.trained_model_dict)
            state_dict_name = 'cache/' + args.trained_model_dict
            state_dict_model = torch.load(state_dict_name)

            model = BertForGraphRetriever.from_pretrained('bert-base-uncased',
                                                          cache_dir=os.path.join("cache/", "transformer_bert/"),
                                                          local_files_only=True,
                                                          state_dict=state_dict_model,
                                                          para_encoder_model=None,
                                                          graph_retriever_config=graph_retriever_config)
        else:
            model = BertForGraphRetriever.from_pretrained('bert-base-uncased',
                                                          cache_dir=os.path.join("cache/", "transformer_bert/"),
                                                          local_files_only=True,
                                                          para_encoder_model=None,
                                                          graph_retriever_config=graph_retriever_config)
    torch.distributed.init_process_group(backend='nccl')

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    POSITIVE = 1.0
    NEGATIVE = 0.0
    num_len_train_features = 81269  # 115078

    # len(train_examples) and len(train_features) are different, by the redundant gold.
    num_train_steps = int(
        num_len_train_features / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_len_train_features)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    model.train()
    torch.backends.cudnn.benchmark = True
    epc = 0
    first = 0

    for epoch_num in range(int(args.num_train_epochs)):
        logger.info('Epoch ' + str(epoch_num + 1))

        # if epoch_num==0:
        #     save(model, "/home/work/user-job-dir/code/", str(-1))
        #     mox.file.copy_parallel(
        #         "/home/work/user-job-dir/code/pytorch_model_" + str(-1) + ".bin",
        #         "s3://obs-app-2019071708472901661/w84184668/hop1_model/" + args.output_dir + "/pytorch_model_" + str(
        #             -1) + ".bin")
        # mox.file.copy_parallel("/home/work/user-job-dir/code/" + args.logfile + '.log',
        #                        "s3://obs-app-2019071708472901661/w84184668/code/" + args.logfile + '.log')

        for fileidx in range(0, 92):
            filename = 'train_file_' + str(fileidx) + '.npy'
            if epoch_num == 0:
                mox.file.copy_parallel(
                    "s3://obs-app-2019071708472901661/w84184668/train_data/hop1_npy_data/" + filename,
                    "/cache/" + filename)
            filename = os.path.join('/cache', filename)

            train_features = np.load_mm(filename)
            train_features = train_features.tolist()

            CHUNK_NUM = 10
            chunk_index = 0

            train_start_index = 0
            TOTAL_NUM = len(train_features)
            current_train_chunk = TOTAL_NUM // CHUNK_NUM

            while train_start_index < TOTAL_NUM:
                train_end_index = min(train_start_index + current_train_chunk - 1, TOTAL_NUM - 1)
                chunk_len = train_end_index - train_start_index + 1

                train_features_ = train_features[train_start_index:train_start_index + chunk_len]
                all_output = torch.tensor([f.output_masks for f in train_features_], dtype=torch.float)
                all_num_paragraphs = torch.tensor([f.num_paragraphs for f in train_features_], dtype=torch.long)
                all_para_contents = torch.tensor([f.para_contents for f in train_features_],
                                                 dtype=torch.long)  # 378 dim
                all_num_ground_triples = torch.tensor([f.p_gts_num[0] for f in train_features_], dtype=torch.long)

                # load the sparse triple matrix for each instance(question+a para)
                all_qt_pts = []
                for f in train_features_:
                    qt_pts = f.qt_pts[0]
                    new_qt_pts = []
                    for i in qt_pts:
                        new_qt_pts.append(i.todense())
                    all_qt_pts.append(new_qt_pts)
                all_qt_pts = np.array(all_qt_pts)
                try:
                    all_qt_pts = torch.tensor(all_qt_pts, dtype=torch.long)
                except:
                    print('when load the data %d from file %s has problem.' % (train_start_index, filename))
                    chunk_index += 1
                    train_start_index = train_end_index + 1
                    continue

                all_qts_num = torch.tensor([f.q_ts_num for f in train_features_], dtype=torch.long)
                all_pts_num = torch.tensor([f.p_ts_num[0] for f in train_features_], dtype=torch.long)

                train_data = TensorDataset(all_output, all_num_paragraphs, all_para_contents, all_qt_pts, all_qts_num,
                                           all_pts_num, all_num_ground_triples)

                train_sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

                tr_loss = 0
                print('Examples from ' + str(train_start_index) + ' to ' + str(train_end_index))
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch_max_len = (batch[4] + batch[5] + batch[
                        6]).max().item()  # 一个batch 内所有para（'q_t+p_t+ground_t'）最大的数据量。
                    num_paragraphs = batch[1]
                    batch_max_para_num = num_paragraphs.max().item()  # 一个batch内 最多的paras 数量。

                    batch = tuple(t.to(device) for t in batch)
                    output_masks, _, para_contents, qt_pts, qt_num, pt_num, gt_num = batch

                    qt_pts = qt_pts[:, :batch_max_para_num, :batch_max_len]
                    para_content = para_contents[:, :batch_max_para_num]
                    gt_num = gt_num[:, :batch_max_para_num]
                    qt_num = qt_num[:]
                    pt_num = pt_num[:, :batch_max_para_num]

                    target = output_masks[:, :batch_max_para_num]

                    if args.negative_num != -1:
                        candidates = np.arange(0, batch_max_para_num).tolist()
                        candidates = random.sample(candidates, args.negative_num)
                        train_indexs = [0, 1, 2] + candidates  # assert ground must covered.
                        train_indexs = list(set(train_indexs))
                    else:
                        train_indexs = np.arange(0, batch_max_para_num).tolist()

                    for train_start in train_indexs:  # short_gold+context:
                        target_ = target[:, train_start:train_start + 1]  # one instance
                        input_qt_pt_ = qt_pts[:, train_start:train_start + 1, :]
                        input_pt_nums = pt_num[:, train_start:train_start + 1]
                        input_qt_nums = qt_num[:]
                        input_gt_nums = gt_num[:, train_start:train_start + 1]
                        input_content = para_content[:, train_start:train_start + 1]

                        loss = model(target_, input_qt_pt_, input_content, input_pt_nums, input_qt_nums, input_gt_nums,
                                     args.max_triple_size,
                                     args.merge_way, args.model_type)

                        if n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        loss.backward()
                        tr_loss += loss.item()
                        max_grad_norm = 1.0

                        # modify learning rate with special warm up BERT uses
                        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        torch.nn.utils.clip_grad_norm(optimizer_grouped_parameters[0]['params'], max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        tb_logger.add_scalar('batch_train_loss',
                                             loss.item(), global_step)
                        print("current loss: %f." % loss.item())

                        global_step += 1

                        torch.cuda.empty_cache()

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
                    #
                    # print('finish train on no%d batch.' % (step))

                chunk_index += 1
                train_start_index = train_end_index + 1

                del all_num_paragraphs
                del all_qt_pts
                gc.collect()
                del all_pts_num
                del all_qts_num
                del train_data
                gc.collect()

            save(model, "/home/work/user-job-dir/code/", str(round(epoch_num + fileidx / 92, 2)))
            mox.file.copy_parallel(
                "/home/work/user-job-dir/code/pytorch_model_" + str(round(epoch_num + fileidx / 92, 2)) + ".bin",
                "s3://obs-app-2019071708472901661/w84184668/hop1_model/" + args.output_dir + "/pytorch_model_" + str(
                    round(epoch_num + fileidx / 92, 2)) + ".bin")
            mox.file.copy_parallel("/home/work/user-job-dir/code/" + args.logfile + '.log',
                                   "s3://obs-app-2019071708472901661/w84184668/code/" + args.logfile + '.log')

            del train_features
            gc.collect()
            # Save the model at the end of the epoch
        save(model, "/home/work/user-job-dir/code/", str(epoch_num + 1))
        mox.file.copy_parallel(
            "/home/work/user-job-dir/code/pytorch_model_" + str(epoch_num + 1) + ".bin",
            "s3://obs-app-2019071708472901661/w84184668/hop1_model/" + args.output_dir + "/pytorch_model_" + str(
                epoch_num + 1) + ".bin")
        mox.file.copy_parallel("/home/work/user-job-dir/code/" + args.logfile + '.log',
                               "s3://obs-app-2019071708472901661/w84184668/code/" + args.logfile + '.log')


if __name__ == '__main__':
    main()
