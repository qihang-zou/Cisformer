# 目前只是测试数据的草案
import torch
import argparse
import os
import scanpy as sc
import tqdm
import resource
import yaml

import sys
sys.path.append("M2Mmodel")
from utils import setup_seed
from utils import PairDataset

def main():
    """_summary_
    The shape and type of saved .pt files:
    Parent (list): rna_sequence, rna_value, atac_sequence, atac_value, enc_pad_mask
    Shape of child (tensor):
    rna_sequence: Cell number * multiple, sequence_len
    rna_value: Cell number * multiple, sequence_len
    atac_sequence: Cell number * multiple, sequence_len, digit(6)
    atac_value: Cell number * multiple, sequence_len
    enc_pad_mask: Cell number * multiple, sequence_len
    """
    setup_seed(2023)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rna", type = str)
    parser.add_argument("-a", "--atac", type = str)
    parser.add_argument("-n", "--name", type = str, default = "preprocessed_data")
    parser.add_argument("-s", "--save_dir", type = str, default = "./")
    parser.add_argument('--atac2rna', action='store_true') #Luz parser.add_argument('--atac2rna', type=bool, default=True)
    parser.add_argument('--dt', type=str) #Luz
    parser.add_argument('--config', type=str) #Luz

    args = parser.parse_args()
    
    rna_dir = args.rna
    atac_dir = args.atac
    file_name = args.name
    save_dir = args.save_dir
    atac2rna = args.atac2rna
    batch = 2 #Luz 10
    dt = args.dt #Luz
    config = args.config

    #Luz config = "atac2rna_config.yaml" if atac2rna else "rna2atac_config.yaml"

    with open(config, "r") as f:
        config = yaml.safe_load(f)
    enc_max_len = config["datapreprocess"]['enc_max_len']
    dec_max_len = config["datapreprocess"]['dec_max_len']
    multiple = config["datapreprocess"]['multiple']
    rna_num_bin = config["model"]['max_express']
    is_raw_count = config["datapreprocess"]['is_raw_count']
    
    os.makedirs(save_dir, exist_ok=True)
    
    atac = sc.read_h5ad(atac_dir)
    rna = sc.read_h5ad(rna_dir)
    dataset = PairDataset(rna, atac, enc_max_len, dec_max_len, rna_num_bin, multiple, atac2rna, is_raw_count) #Luz dataset = PairDataset(rna.X, atac.X, enc_max_len, dec_max_len, rna_num_bin, multiple, atac2rna, is_raw_count)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch, num_workers = 20)
        
    i = 1
    num = 0
    out_rna_sequence = []
    out_rna_value = []
    out_atac_sequence = []
    out_atac_value = []
    out_enc_pad_mask = []
    
    for n, (rna_sequence, rna_value, atac_sequence, atac_value, enc_pad_mask) in enumerate(tqdm.tqdm(dataloader, ncols=80)): #Luz for n, (rna_sequence, rna_value, atac_sequence, atac_value, enc_pad_mask) in enumerate(tqdm.tqdm(dataloader)):
        # rna_sequence: b, mul, n
        # rna_value: b, mul, n
        # atac_sequence: b, n, mul, 6
        # atac_value: b, mul, n
        # enc_pad_mask: b, mul, n
        
        out_rna_sequence.append(rna_sequence.view(-1, rna_sequence.shape[-1]))
        out_rna_value.append(rna_value.view(-1, rna_value.shape[-1]))
        out_atac_sequence.append(atac_sequence.view(-1, atac_sequence.shape[-2], atac_sequence.shape[-1]))
        out_atac_value.append(atac_value.view(-1, atac_value.shape[-1]))
        out_enc_pad_mask.append(enc_pad_mask.view(-1, enc_pad_mask.shape[-1]))
        
        #Luz if i:
        #Luz     # 获取当前进程的内存占用（在Unix系统上可用）
        #Luz     memory_info = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        #Luz     print(f"\nMemory used: {memory_info * 10 / 1024:.2f} MB\n")
        #Luz     i = 0
        
        if dt!='test':
            cnt = 8000
        else:
            cnt = 40

        if ((n+1) * multiple * batch % cnt)==0: #if n * multiple * batch >= cnt and n * multiple * batch % cnt < multiple: #Luz if n * multiple * batch >= 10000 and n * multiple * batch % 10000 < multiple:
            torch.save([
                        torch.cat(out_rna_sequence, dim = 0), 
                        torch.cat(out_rna_value, dim = 0), 
                        torch.cat(out_atac_sequence, dim = 0), 
                        torch.cat(out_atac_value, dim = 0),
                        torch.cat(out_enc_pad_mask, dim = 0)
                        ], 
                    os.path.join(save_dir, file_name + "_" + str(num) + ".pt"))
            out_rna_sequence = []
            out_rna_value = []
            out_atac_sequence = []
            out_atac_value = []
            out_enc_pad_mask = []  #Luz!!!!!!!!!!!!!!
            num += 1
            
    if out_rna_sequence != []:
        torch.save([
                    torch.cat(out_rna_sequence, dim = 0), 
                    torch.cat(out_rna_value, dim = 0), 
                    torch.cat(out_atac_sequence, dim = 0), 
                    torch.cat(out_atac_value, dim = 0),
                    torch.cat(out_enc_pad_mask, dim = 0)
                    ], 
                os.path.join(save_dir, file_name + "_" + str(num) + ".pt"))
       
if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
