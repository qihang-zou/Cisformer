import warnings
warnings.filterwarnings("ignore")
import torch
# import argparse
import os
import scanpy as sc
import tqdm
# import resource
import yaml
import pandas as pd
import random

# import sys
# sys.path.append("M2Mmodel")
from cisformer.map_rna_atac import main as scmap
from cisformer.M2Mmodel.utils import setup_seed, PairDataset
torch.multiprocessing.set_sharing_strategy('file_system')

def pair_data_process(rna, 
                      atac, 
                      enc_max_len, 
                      dec_max_len, 
                      rna_num_bin, 
                      multiple, 
                      atac2rna, 
                      batch,
                      num_workers,
                      shuffle,
                      cnt,
                      save_dir,
                      is_raw_count,
                      file_name
                      ):
    os.makedirs(save_dir, exist_ok=True)
    tag = os.path.split(save_dir)[1]
    dataset = PairDataset(rna, atac, enc_max_len, dec_max_len, rna_num_bin, multiple, atac2rna, is_raw_count)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch, num_workers = num_workers, shuffle=shuffle)
    # file_name = str(enc_max_len) + "_" + str(dec_max_len)
    
    if not shuffle :
        obs = pd.merge(rna.obs, atac.obs, left_index=True, right_index=True, suffixes=("_rna", "_atac"))
        obs.to_csv(os.path.join(save_dir, "cell_info.tsv"), sep="\t")
    
    i = 1
    num = 0
    out_rna_sequence = []
    out_rna_value = []
    out_atac_sequence = []
    out_atac_value = []
    out_enc_pad_mask = []
    
    for rna_sequence, rna_value, atac_sequence, atac_value, enc_pad_mask in tqdm.tqdm(dataloader, ncols=80, desc=f"Generating {tag}:"):
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
        
        # if i:
        #     # 获取当前进程的内存占用（在Unix系统上可用）
        #     memory_info = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        #     print(f"\nMemory used per process: {memory_info * 10 / 1024:.2f} MB\n")
        #     i = 0
        
        if len(out_rna_sequence) * batch * multiple >= cnt:
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
            out_enc_pad_mask = []
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


def main(rna, atac, manually, atac2rna, save_dir, config, batch_size, num_workers, cnt, shuffle, dec_whole_length):
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
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-r", "--rna", type = str, help="Should be .h5ad format")
    # parser.add_argument("-a", "--atac", type = str, help="Should be .h5ad format")
    # parser.add_argument('-c', '--config', type = str, help = "Config file") 
    # parser.add_argument("-s", "--save_dir", type = str, default = "./")
    # parser.add_argument('--cnt', type = int, default = 10000, help = "The number of cell will be contained per output files") 
    # parser.add_argument('--batch_size', type = int, default = 10, help = "The larger, the faster, but will increase memory usage")
    # parser.add_argument('--num_workers', type = int, default = 10, help = "The number of processors. The larger, the faster, but will increase memory usage")
    # parser.add_argument('--atac2rna', action = 'store_true')
    # parser.add_argument('--manually', action = 'store_true', help = 'By default the script will divide rna and atac into train, val and test set. If you set the parameter, this script will not divide the input dataset.')
    # parser.add_argument('--shuffle', action = 'store_true', help = "This is used for training dataset. For testing, you should set this to False and you will also got cell_info file for testing dataset. This is used only when 'manually' is set. ")
    # parser.add_argument('--dec_whole_length', action = 'store_true', help = "If set, the decode modality will use the whole length. This is used only when 'manually' is set.")
   
    # args = parser.parse_args()
    
    rna_dir = rna
    atac_dir = atac
    manually = manually
    atac2rna = atac2rna
    save_dir = save_dir
    if config:
        config = config
    else:
        config = os.path.join("cisformer_config","atac2rna_config.yaml") if atac2rna else os.path.join("cisformer_config","rna2atac_config.yaml")
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    log1p = True if atac2rna else False
    
    pair_args = { 
        "rna_num_bin": config["model"]['max_express'], 
        "atac2rna": atac2rna, 
        "batch": batch_size,
        "num_workers": num_workers,
        "is_raw_count": False if atac2rna else True
    }
    
    os.makedirs(save_dir, exist_ok=True)
    _, rna_name = os.path.split(rna_dir)
    _, atac_name = os.path.split(atac_dir)
    rna_name = rna_name.replace(".h5ad","")
    atac_name = atac_name.replace(".h5ad","")
    file_name = atac_name+"2"+rna_name if atac2rna else rna_name+"2"+atac_name
    if not manually:
        mapped_dict = scmap(rna_dir, atac_dir, save_dir, divide=True, log1p=log1p)
        train_rna = mapped_dict['train_rna']
        train_atac = mapped_dict['train_atac']
        test_rna = mapped_dict['test_rna']
        test_atac = mapped_dict['test_atac']
        
        if atac2rna:
            pair_data_process(train_rna, 
                            train_atac,
                            enc_max_len = config["datapreprocess"]['enc_max_len'],
                            dec_max_len = config["datapreprocess"]['dec_max_len'], 
                            shuffle = True, 
                            save_dir = os.path.join(save_dir, "cisformer_atac2rna_train_dataset"),
                            cnt= cnt,
                            multiple = config["datapreprocess"]['multiple'], 
                            file_name = file_name,
                            **pair_args)
            pair_data_process(test_rna, 
                              test_atac, 
                              enc_max_len = config["datapreprocess"]['enc_max_len'],
                              dec_max_len = config["datapreprocess"]['dec_max_len'], 
                              shuffle = False, 
                              save_dir = os.path.join(save_dir, "cisformer_atac2rna_test_dataset"),
                              cnt= float('inf'),
                              multiple = 1,
                              file_name = file_name,
                              **pair_args)
        else:
            # cells = list(train_rna.obs_names)
            # train_set = random.sample(cells, int(len(cells) * 0.8))
            # val_set = [each for each in cells if each not in train_set]
            # new_train_rna = train_rna[train_set, :]
            # new_train_atac = train_atac[train_set, :]
            # val_rna = train_rna[val_set, :]
            # val_atac = train_atac[val_set, :]
            pair_data_process(train_rna, 
                            train_atac,
                            enc_max_len = config["datapreprocess"]['enc_max_len'],
                            dec_max_len = config["datapreprocess"]['dec_max_len'], 
                            shuffle = True, 
                            save_dir = os.path.join(save_dir, "cisformer_rna2atac_train_dataset"),
                            cnt= cnt,
                            multiple = config["datapreprocess"]['multiple'], 
                            file_name = file_name,
                            **pair_args)
            pair_data_process(test_rna, 
                            test_atac,
                            enc_max_len = config["datapreprocess"]['enc_max_len'],
                            dec_max_len = config["datapreprocess"]['dec_max_len'], 
                            shuffle = True, 
                            save_dir = os.path.join(save_dir, "cisformer_rna2atac_val_dataset"),
                            cnt= cnt,
                            multiple = 1, 
                            file_name = file_name,
                            **pair_args)
            # pair_data_process(test_rna, 
            #                   test_atac, 
            #                   enc_max_len = config["datapreprocess"]['enc_max_len'],
            #                   dec_max_len = "whole", 
            #                   shuffle = False, 
            #                   save_dir = os.path.join(save_dir, "cisformer_rna2atac_test_dataset"),
            #                   cnt= float('inf'),
            #                   multiple = 1,
            #                   file_name = file_name,
            #                   **pair_args)
        
    else:
        mapped_dict = scmap(rna_dir, atac_dir, save_dir, divide=False, log1p=log1p)
        rna = mapped_dict['rna_new']
        atac = mapped_dict['atac_new']
        pair_data_process(rna, 
                          atac,
                          enc_max_len = config["datapreprocess"]['enc_max_len'], 
                          dec_max_len = config["datapreprocess"]['dec_max_len'] if not dec_whole_length else "whole", 
                          shuffle = shuffle, 
                          save_dir = save_dir,
                          cnt= cnt,
                          multiple = config["datapreprocess"]['multiple'],
                          file_name = file_name,
                          **pair_args)
        
if __name__ == "__main__":
    main()