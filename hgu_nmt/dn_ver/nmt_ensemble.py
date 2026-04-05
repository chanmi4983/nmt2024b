# Referenced from http://github.com/pytorch/fairseq/scripts/average_checkpoints.py

import argparse
import collections
import os

import torch


def Ensemble(args):
    subdir = args.save_dir + args.subdir + '/'
    
    #num_ck = args.n_checkpoint_ensemble

    #model_file_list = [f for f in os.listdir(subdir) if f[:5] == 'model' and f[-4:] == '.pth'\
    #                                                    and f[-9:] != '.best.pth']
    model_file_list = [f for f in os.listdir(subdir) if f[:5] == 'model' and f[-4:] == '.pth']
    print("MODEL FILE LIST : ", model_file_list)
    num_ck = len(model_file_list)

    #return 0
    #max_num = 0
    #for file_name in model_file_list:
    #    model_num = int(file_name.strip('model').strip('.pth'))
    #    if model_num > max_num:
    #        max_num = model_num

    params_dict = collections.OrderedDict()

    #for i in range(max_num-num_ck+1,max_num+1):
    for tmp_model_path in model_file_list:
        tmp_model_path = args.save_dir + args.subdir + '/' + tmp_model_path
        if os.path.exists(tmp_model_path):
            chk_point = torch.load(tmp_model_path, map_location='cpu')
            tmp_state = chk_point['state_dict']
        else:
            raise KeyError("Checkpoint File is missing {}".format(tmp_model_path))

        params_keys = list(tmp_state.keys())
        for k in params_keys:
            p = tmp_state[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_ck)
        else:
            averaged_params[k] //= num_ck

    save_list = {'iloop':None, 'state_dict':averaged_params, 'scheduler':None, 'optimizer':None}

    #torch.save(save_list, args.save_dir + args.subdir + '/model.avg')
    return save_list
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    # Files
    parser.add_argument("--save_dir", type=str, default='./results/')
    parser.add_argument("--subdir", type=str, default='de2en_tm_1join_0.1smooth_0.1drop_')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    Ensemble(args)
