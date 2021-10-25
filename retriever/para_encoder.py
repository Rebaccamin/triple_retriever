import os
#os.system("pip install tqdm")
##os.system("pip install transformers")
#os.system("pip install torch")
#os.system("pip install scipy")
#os.system("pip install tqdm")


from transformers import BertPreTrainedModel, BertModel, BertTokenizer
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import argparse
from utils_triple import convert_examples_to_features_text
#import moxing as mox

import pickle
import random
import numpy as np
from transformers import AdamW,get_linear_schedule_with_warmup
from utils_triple import warmup_linear,save
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import logging
import gc
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Para_encoder(BertPreTrainedModel):
    def __init__(self,config):
        super(Para_encoder, self).__init__(config)
        config.output_hidden_states = True

        self.bert = BertModel(config)
        self.apply(self._init_weights)
        self.cpu = torch.device('cpu')

    def forward(self,input_question,input_paras,target):
        B = input_question.size(0)
        N = input_paras.size(1)
        L = input_paras.size(2)


        encoded_layers = self.bert(input_question)
        question_triple_embeds = encoded_layers[0][:, 0] #B  768

        input_paras = input_paras.contiguous().view(B * N, L)
        encoded_layers = self.bert(input_paras)
        para_triple_embeds = encoded_layers[0][:, 0] #B*N 768

        inner_product_score=torch.mm(question_triple_embeds,para_triple_embeds.transpose(0, 1))# B B*N

        index_0=torch.arange(B*N)
        index_0=index_0.view(B,N)
        index_0=index_0.to(inner_product_score.device)

        output=torch.gather(inner_product_score,dim=1,index=index_0)#B x N
        output=output.to(target.device)

        loss = F.binary_cross_entropy_with_logits(output, target, reduction='mean')
        return loss

def read(file):
    with open(file,'rb')as f:
        data=pickle.load(f)
    return data

def write(file,data):
    true_file=file.replace('.pkl','.npy')
    true_file=true_file.split('/')[-1]
    np.save("/home/work/user-job-dir/code/"+true_file,data)

    mox.file.copy_parallel(
        "/home/work/user-job-dir/code/"+true_file,
        "s3://obs-app-2019071708472901661/w84184668/train_data/pickle_dir3_para_encoder_npy/" + true_file)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size',default=8,type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--gradient_accumulation_steps',default=1)
    parser.add_argument('--num_train_epochs',default=3,type=int)
    parser.add_argument('--learning_rate',default=3e-5,
                        type=float)
    parser.add_argument('--warmup_proportion',default=0.1,type=float)
    args=parser.parse_args()


    max_seq_length=378
    mox.file.copy_parallel(
        "s3://obs-app-2019071708472901661/w84184668/train_data/pickle_dir3_para_encoder_npy/",
        "/cache")

    location_dir = mox.file.list_directory("/cache", recursive=True)
    # location_dir='data/pickle_dir3'
    train_features = []
    train_features_files = []
    for idx, location in enumerate(location_dir):
        if '.npy' in location :
            train_features_files.append(os.path.join('/cache', location))
    print('***load %d files.***'%len(train_features_files))

    # for dir,_,files in os.walk(location_dir):
    #     for file in files:
    #         train_features_files.append(os.path.join(dir,file))
    # print('load %d pkl files.'%len(train_features_files))

    mox.file.copy_parallel("s3://obs-app-2019071708472901661/w84184668/code/transformers_tokenizer/",
                           "cache/transformer_tokenizer/")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              cache_dir=os.path.join("cache/", "transformer_tokenizer/"),
                                              do_lower_case=True, local_files_only=True)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
    #                                           cache_dir="transformer_tokenizer/",
    #                                           do_lower_case=True, local_files_only=True)
    # for file in train_features_files:
    #     examples=read(file)
    #     print('start process file %s.'%file)
    #     temp_data=convert_examples_to_features_text(examples,max_seq_length, tokenizer,max_size=50)
    #     train_features += temp_data
    #     write(file,temp_data)
    #     del temp_data
    #     gc.collect()
    # np.save('debug.npy',train_features)
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    train_features=[]
    for file in train_features_files:
       train_features+=np.load(file).tolist()

    mox.file.copy_parallel(
        "s3://obs-app-2019071708472901661/w84184668/code/bert_model/",
        "cache/transformer_bert/")
    model = Para_encoder.from_pretrained('bert-base-uncased',cache_dir=os.path.join("cache/", "transformer_bert/"),
                                                  local_files_only=True)

    # model = Para_encoder.from_pretrained('bert-base-uncased', cache_dir="transformer_bert",
    #                                      local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    num_len_train_features = len(train_features)
    num_train_steps = int(
        num_len_train_features / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

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
    for _ in range(int(args.num_train_epochs)):
        logger.info('Epoch ' + str(epc + 1))
        CHUNK_NUM = 10
        chunk_index = 0
        TOTAL_NUM = len(train_features)
        current_train_chunk = TOTAL_NUM // CHUNK_NUM
        train_start_index = 0

        while train_start_index < TOTAL_NUM:

            train_end_index = min(train_start_index + current_train_chunk - 1, TOTAL_NUM - 1)
            chunk_len = train_end_index - train_start_index + 1

            train_features_ = train_features[train_start_index:train_start_index + chunk_len]
            questions = torch.tensor([f.q_embed for f in train_features_], dtype=torch.long)
            paragraphs = torch.tensor([f.p_embeds for f in train_features_], dtype=torch.long)

            target = torch.tensor([f.label for f in train_features_], dtype=torch.float)
            # except:
            #     for f in train_features_:
            #         temp=torch.tensor(f.label)
            #         print('*'*20)
            # train_start_index=train_end_index+1
            # continue

            train_data = TensorDataset(questions,paragraphs,target)

            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            tr_loss = 0

            logger.info('Examples from ' + str(train_start_index) + ' to ' + str(train_end_index))
            train_start=0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                q,ps,y=batch
                B = q.size(0)
                loss=model(q,ps,y)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                # if args.gradient_train_startaccumulation_steps > 1:
                #     loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                train_start = train_start + 1

                max_grad_norm = 1.0

                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                torch.nn.utils.clip_grad_norm(optimizer_grouped_parameters[0]['params'], max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                del q
                del ps
            #del target

            train_start_index = train_end_index + 1
            chunk_index=chunk_index+1


        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join("/home/work/user-job-dir/code/", "para_encoder_model_" + str(epc) + ".bin")

        try:
            torch.save(model_to_save.state_dict(), output_model_file)
            mox.file.copy_parallel(
                "/home/work/user-job-dir/code/para_encoder_model_" + str(epc) + ".bin",
                "s3://obs-app-2019071708472901661/w84184668/para_encoder_model"+"/para_encoder_model_" + str(
                    epc) + ".bin")
        except:
            print('cannot save model.....')

        epc=epc+1


if __name__ == '__main__':
    main()

