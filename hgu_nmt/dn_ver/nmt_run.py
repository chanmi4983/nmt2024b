# H. Choi, hchoi@handong.edu
# DongNyeong Heo, sjglsks@gmail.com worked for DistributedDataParallel

import os
import argparse
import pickle as pkl

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from libs.utils import make_subdir, CountDown
from dataset import CallNormalIterator_TokenBased, CallTestIterator_TokenBased,\
                    CallTestIterator_SacreBLEU
from nmt_main import train_main
from nmt_model_tm import NormalNMT
from nmt_trans import Test_SacreBLEU

import copy

neptune_project='' # Neptune project name

def train_run():
    rank = args.local_rank
    if rank == 0:
        print(args)
        pkl.dump(args, open(args.model_file+'.args.pkl', 'wb'), -1)
        with open(args.model_file+'.args', 'w') as fp:
            for key in vars(args):
                fp.write(key + ': ' + str(getattr(args, key)) + '\n')

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(rank)

    train_iter = CallNormalIterator_TokenBased(args.data_dir, args.train_src_file,\
                        args.train_trg_file,\
                        args.src_dict, args.trg_dict, args.token_size,\
                        args.ahead, seed=args.dataset_seed,\
                        rank=rank, world_size=args.world_size, sorting=bool(args.sorting))
    valid_iter = CallTestIterator_TokenBased(args.data_dir, args.valid_src_file,\
                        args.valid_trg_file,\
                        args.src_dict, args.trg_dict, args.test_token_size,\
                        rank=rank, world_size=args.world_size, sorting=True)
    args.src_words_n = len(train_iter.vocab_dict['src'].keys())
    args.trg_words_n = len(train_iter.vocab_dict['trg'].keys())

    print("Training..")
    train_main(args, train_iter, valid_iter, rank, neptune_project)

def test_run():
    use_cuda = torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")

    test_args_file = args.save_dir + args.test_subdir + '/' + args.test_model_args
    old_args = pkl.load(open(test_args_file, 'rb'))

    args.src_dict = old_args.src_dict
    args.trg_dict = old_args.trg_dict

    test_iter = CallTestIterator_SacreBLEU(args.data_dir, args.test_src_file, args.test_trg_file,\
                            args.test_ref_src_file, args.test_ref_trg_file,\
                            args.test_token_ref_src_file, args.test_token_ref_trg_file,\
                            args.src_dict, args.trg_dict, args.test_batch_size)
    old_args.max_length = 250

    old_args.src_words_n = len(test_iter.dataset.src_dict2.keys())
    old_args.trg_words_n = len(test_iter.dataset.trg_dict2.keys())

    old_args.model_file = args.save_dir + args.test_subdir + '/model'
    if args.test_model_file == '':
        args.test_model_file = args.save_dir + args.test_subdir + '/' + args.src_lang+'2'+args.trg_lang + '.ensemble_model.best.pth'
    else:
        args.test_model_file = args.save_dir + args.test_subdir + '/' + args.test_model_file
    print("model load: " + args.test_model_file)
    print("test data : ", args.test_src_file)

    model = NormalNMT(args=old_args).to(device)

    chk_point = torch.load(args.test_model_file)
    model.load_state_dict(chk_point['state_dict'], strict=False)

    print("# Parameters : ", sum( p.numel() for p in model.parameters() if p.requires_grad ) )

    old_args.beam_width = args.beam_width
    old_args.sacrebleu_tokenizer = args.sacrebleu_tokenizer
    old_args.sacrebleu_lowercase = args.sacrebleu_lowercase

    print("Testing...")
    Test_SacreBLEU(model, test_iter, old_args)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--port", type=int, default=24378)
    parser.add_argument("--translation_task", type=str, default='wmt14')
    parser.add_argument("--mode", type=str, default='train') # 'train', 'test', 'translate'
    
    # Datasets
    parser.add_argument("--src_lang", type=str, default='')
    parser.add_argument("--trg_lang", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--train_src_file", type=str, default='')
    parser.add_argument("--train_trg_file", type=str, default='')
    parser.add_argument("--valid_src_file", type=str, default='')
    parser.add_argument("--valid_trg_file", type=str, default='')
    parser.add_argument("--test_src_file", type=str, default='')
    parser.add_argument("--test_trg_file", type=str, default='')
    parser.add_argument("--test_ref_src_file", type=str, default='')
    parser.add_argument("--test_ref_trg_file", type=str, default='')
    parser.add_argument("--test_token_ref_src_file", type=str, default='')
    parser.add_argument("--test_token_ref_trg_file", type=str, default='')
    parser.add_argument("--src_dict", type=str, default='')
    parser.add_argument("--trg_dict", type=str, default='')
    parser.add_argument("--max_length", type=int, default=250)
    parser.add_argument("--bleu_script", type=str, default='./tools/multi-bleu.perl')
    parser.add_argument("--ahead", type=int, default=18000)
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    # --- For token-based dataloader
    parser.add_argument("--token_size", type=int, default=7600)
    parser.add_argument("--test_token_size", type=int, default=7600)
    parser.add_argument("--sorting", type=int, default=0)

    # training
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--opt_start", type=int, default=4000)
    parser.add_argument("--opt_scheduled", type=int, default=0)
    parser.add_argument("--adam_beta_max", type=float, default=0.98)
    parser.add_argument("--adam_eps", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--update_step", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--valid_start", type=int, default=50)
    parser.add_argument("--valid_every", type=int, default=50)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--n_checkpoint", type=int, default=8) # 8 -> store 10 checkpoints
    parser.add_argument("--patience", type=int, default=20)

    # RESUME & Loading
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--load_neptune_exp_id", type=str, default='')

    # Model architecture
    parser.add_argument("--model", type=str, default='nmt')
    parser.add_argument("--joined_dictionary", type=int, default=1)
    parser.add_argument("--dim_wemb", type=int, default=512)
    parser.add_argument("--dim_model", type=int, default=512)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--emb_noise", type=float, default=0.0)
    ### RNNNMT 
    parser.add_argument("--rnn_name", type=str, default='lstm')
    parser.add_argument("--dim_enc", type=int, default=0)
    parser.add_argument("--dim_att", type=int, default=0)
    parser.add_argument("--pos_enc", type=int, default=1)
    ### Transformer 
    parser.add_argument("--tm_n_layers", type=int, default=6)
    parser.add_argument("--tm_dim_ff", type=int, default=2048) 
    parser.add_argument("--tm_n_head", type=int, default=8)
    parser.add_argument("--tm_dk", type=int, default=64)
    parser.add_argument("--tm_dv", type=int, default=64)

    # Testing
    parser.add_argument("--test_subdir", type=str, default='')
    parser.add_argument("--test_model_file", type=str, default='')
    parser.add_argument("--test_model_args", type=str, default='')
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--sacrebleu_tokenizer", type=str, default='13a')
    parser.add_argument("--sacrebleu_lowercase", type=int, default=0)

    # ETC
    parser.add_argument("--neptune", type=int, default=0)

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.mode == 'train':
        # Add any hyperparameter for subdir name as tuple
        args.subdir, args.model_file = make_subdir((args.translation_task, ''),\
                            (args.src_lang+'2'+args.trg_lang, ''),\
                            (args.model, ''),\
                            (args.lr, 'lr'),\
                            (args.token_size, 'TK'),\
                            (args.update_step, 'US'),\
                            (args.sorting, 'sort'),\
                            (args.patience, 'pat'),\
                            (args.dropout_p, 'drop'),\
                            (args.tm_n_layers, 'layer'),\
                            (args.dim_model, 'D'),\
                            (args.tm_n_head, 'H'),\
                            save_dir=args.save_dir, resume=args.resume, rank=args.local_rank)
        train_run()
    elif args.mode == 'test':
        test_run()
