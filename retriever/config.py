import argparse


def train_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--max_triple_size", default=20, type=int)
    parser.add_argument('--data_url', type=str, default=None,
                        help='the input data dir. sepecific for huawei server. modelarts.')
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument('--output_dir', type=str, default='trained_model')
    parser.add_argument('--task',
                        type=str,
                        default='hotpot_open',
                        required=False,
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
                        default=2,
                        type=int,
                        help="Total batch size for training.")
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

    parser.add_argument("--cuda_device", default=7, type=int)

    parser.add_argument('--file_start_idx', default=2, type=int)
    parser.add_argument('--train_url', type=str, default=None)
    parser.add_argument('--negative_num', type=int, default=-1)
    parser.add_argument('--para_encoder', action='store_true')
    parser.add_argument('--merge_way', default='concatenate', help='concatenate,add')
    parser.add_argument('--model_type',default='triple_only',type=str,help='triple_only, ')
    parser.add_argument('--logfile',default='train_log',type=str)
    parser.add_argument('--continue_train',action="store_true")
    parser.add_argument('--trained_model_dict',type=str,default='')
    parser.add_argument('--metrics', default='max', type=str, help='mean,max')

    parser.add_argument('--rank', type=int, default=0, help='Index of current task')  # 表示当前是第几个节点
    parser.add_argument('--world_size', type=int, default=1, help='Total number of tasks')  # 表示一共有几个节点
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='distributed backend')
    parser.add_argument('--init_method', default=None,
                        help='print process when training')
    parser.add_argument('--local_rank')





    args = parser.parse_args()
    return args