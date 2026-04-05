import torch
import argparse

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--file", type=str, default='')
args = parser.parse_args()

# Model loading
print('results/'+args.file)
model = torch.load('results/' +args.file)
param = model['state_dict']

if 0:
    encoder_ln = [torch.mean(param['encoder.layer_stack.0.self_attn.layer_norm.weight']), 
        torch.mean(param['encoder.layer_stack.1.self_attn.layer_norm.weight']), 
        torch.mean(param['encoder.layer_stack.2.self_attn.layer_norm.weight']), 
        torch.mean(param['encoder.layer_stack.3.self_attn.layer_norm.weight']), 
        torch.mean(param['encoder.layer_stack.4.self_attn.layer_norm.weight']), 
        torch.mean(param['encoder.layer_stack.5.self_attn.layer_norm.weight'])]

    decoder_ln = [torch.mean(param['decoder.layer_stack.0.self_attn.layer_norm.weight']), 
        torch.mean(param['decoder.layer_stack.1.self_attn.layer_norm.weight']), 
        torch.mean(param['decoder.layer_stack.2.self_attn.layer_norm.weight']), 
        torch.mean(param['decoder.layer_stack.3.self_attn.layer_norm.weight']), 
        torch.mean(param['decoder.layer_stack.4.self_attn.layer_norm.weight']), 
        torch.mean(param['decoder.layer_stack.5.self_attn.layer_norm.weight'])]

    e_ = [t.item() for t in encoder_ln]
    d_ = [t.item() for t in decoder_ln]
    print('Enc', e_)
    print('dec', d_)

if 1:
    if 'encoder.layer_stack.0.self_attn.sdp_attn.temper' not in param.keys():
        print('temper is not trainable')
        exit()

    encoder_temp = [param['encoder.layer_stack.0.self_attn.sdp_attn.temper'],
        param['encoder.layer_stack.1.self_attn.sdp_attn.temper'], 
        param['encoder.layer_stack.2.self_attn.sdp_attn.temper'], 
        param['encoder.layer_stack.3.self_attn.sdp_attn.temper'], 
        param['encoder.layer_stack.4.self_attn.sdp_attn.temper'],
        param['encoder.layer_stack.5.self_attn.sdp_attn.temper']]

    decoder_temp = [param['decoder.layer_stack.0.self_attn.sdp_attn.temper'],
        param['decoder.layer_stack.1.self_attn.sdp_attn.temper'], 
        param['decoder.layer_stack.2.self_attn.sdp_attn.temper'], 
        param['decoder.layer_stack.3.self_attn.sdp_attn.temper'], 
        param['decoder.layer_stack.4.self_attn.sdp_attn.temper'],
        param['decoder.layer_stack.5.self_attn.sdp_attn.temper']]

    enc_dec_temp = [param['decoder.layer_stack.0.enc_attn.sdp_attn.temper'],
        param['decoder.layer_stack.1.enc_attn.sdp_attn.temper'], 
        param['decoder.layer_stack.2.enc_attn.sdp_attn.temper'], 
        param['decoder.layer_stack.3.enc_attn.sdp_attn.temper'], 
        param['decoder.layer_stack.4.enc_attn.sdp_attn.temper'],
        param['decoder.layer_stack.5.enc_attn.sdp_attn.temper']]


    e_temp = [t.item() for t in encoder_temp]
    d_temp = [t.item() for t in decoder_temp]
    ed_temp = [t.item() for t in enc_dec_temp]


    print('Enc', e_temp)
    print('Dec', d_temp)
    print('E-D', ed_temp)
