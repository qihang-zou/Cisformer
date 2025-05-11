import warnings
warnings.filterwarnings("ignore")
# import sys
import tqdm
# import argparse
import os
import yaml
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
import pandas as pd
from importlib.resources import files as rfiles

import torch
# sys.path.append("M2Mmodel")
from cisformer.M2Mmodel.M2M import M2M_atac2rna
from cisformer.M2Mmodel.utils import *



def main(data, output, load, config_file, name):
    
    # parser = argparse.ArgumentParser(description='atac2rna predict script of cisformer')
    
    # # path
    # parser.add_argument('-d', '--data', type=str, help="Preprocessed file end with '.pt'. You can only specify one file!")
    # parser.add_argument('-n', '--name', type=str, help='Name of the output file. Without .h5ad')
    # parser.add_argument('-o', '--output', type=str, default="output", help="output dir")
    # parser.add_argument('-c', '--config_file', type=str, default="config/atac2rna_config.yaml", help="config file for model parameters")
    # parser.add_argument('-l', '--load', type=str, default=None, help="Load previous model, path to previous model state_dict")
    # # parser.add_argument('--raw_rna', type=str, default=None, help='Path to the target RNA h5ad. This is used to transfer the vars and obs from raw RNA to predict file')
    
    # args = parser.parse_args()
    
    # parameters
    data_dir = data
    output_dir = output
    saved_model_path = load
    config_file = config_file
    # raw_rna = args.raw_rna
    name = name
    
    if os.path.isdir(data_dir):
        files = os.listdir(data_dir)
    else:
        data_dir, files = os.path.split(data_dir)
        files = [files]
    files = [f for f in files if f.endswith(".pt")]
    assert len(files) > 0, "No file with format '.pt'"
    ##################################sort
    files = pd.DataFrame([each.split("_") for each in files])
    files.iloc[:, -1] = files.iloc[:, -1].map(lambda x: x[:-3]).astype(int)
    files = files.sort_values(files.columns[-1])
    files = files.astype(str)
    files = ["_".join(each.to_list()) + ".pt" for _, each in files.iterrows()]
    
    # assert not os.path.isdir(file), f"'{file}' is not a file"
    obs = pd.read_csv(os.path.join(data_dir, "cell_info.tsv"), sep="\t", index_col=0)
    gene = pd.read_table(rfiles("cisformer.resource")/'human_genes.tsv', names=['gene_ids', 'gene_name'])
    var = pd.DataFrame(index = gene['gene_name'])
    
    # from config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    batch_size = config["training"].get('batch_size')
    SEED = config["training"].get('SEED')
    num_workers = config["training"].get('num_workers')

    # init
    setup_seed(SEED)
    # accelerator = New_Accelerator(step_scheduler_with_optimizer = False)
    # device = accelerator.device
    device = select_least_used_gpu()
    
    # parameters for dataloader construction
    dataloader_kwargs = {'batch_size': batch_size, 'shuffle': False}
    cuda_kwargs = {
        # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
        "num_workers": num_workers,
        'prefetch_factor': 2
    }  
    dataloader_kwargs.update(cuda_kwargs)
    
    # with accelerator.main_process_first():    
    # model loading
    model = M2M_atac2rna(
        dim = config["model"].get("dim"),
        
        enc_depth = 0,
        enc_heads = config["model"].get("dec_heads"),
        enc_ff_mult = config["model"].get("dec_ff_mult"),
        enc_dim_head = config["model"].get("dec_dim_head"),
        enc_emb_dropout = config["model"].get("dec_emb_dropout"),
        enc_ff_dropout = config["model"].get("dec_ff_dropout"),
        enc_attn_dropout = config["model"].get("dec_attn_dropout"),
        
        dec_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
        dec_num_value_tokens = config["model"].get('max_express') + 1, # +1 for 0
        dec_depth = config["model"].get("dec_depth"),
        dec_heads = config["model"].get("dec_heads"),
        dec_ff_mult = config["model"].get("dec_ff_mult"),
        dec_dim_head = config["model"].get("dec_dim_head"),
        dec_emb_dropout = config["model"].get("dec_emb_dropout"),
        dec_ff_dropout = config["model"].get("dec_ff_dropout"),
        dec_attn_dropout = config["model"].get("dec_attn_dropout")
    )
    # model = model.half() #将模型参数类型改为FP16，半精度浮点数
    
    # 加载模型
    if saved_model_path:
        model.load_state_dict(torch.load(saved_model_path), strict = True)
        # accelerator.print("previous model loaded!")
        print("previous model loaded!")
    else:
        # accelerator.print("warning: you haven't load previous model!")
        print("warning: you haven't load previous model!")
       
    # model, loader = accelerator.prepare(model, loader)
    model = model.half()
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)
    output_rna = []
    for file in files:     
        data = torch.load(os.path.join(data_dir, file))
        dataset = PreDataset(data)
        loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

        predict_express = []
        predict_sequence = []
        nonpad_idx = []
        model.eval()
        with torch.no_grad():  
            # data_iter = tqdm.tqdm(loader, total=len(loader), desc="predicting", ncols=80, bar_format='{l_bar}{bar}| {percentage:.0f}%') if accelerator.is_main_process else loader
            # for inputsrna_sequence, rna_value, atac_sequence, _, enc_pad_mask in data_iter:
            data_iter = tqdm.tqdm(loader, total=len(loader), desc="predicting", ncols=80, bar_format='{l_bar}{bar}| {percentage:.0f}%')
            for inputs in data_iter:
                rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
                # rna_sequence: b, n
                # rna_value: b, n
                # atac_sequence: b, n, 6
                # enc_pad_mask: b, n
                
                # run model
                logist = model(atac_sequence, rna_sequence, enc_mask = enc_pad_mask, dec_mask = (rna_value != 0)) # b, n, num_value_tokens
                logist_hat = logist.argmax(dim = -1) # b, n
                # predict_value = accelerator.gather_for_metrics(logist_hat)
                predict_value = logist_hat
                predict_express.append(predict_value)
                # predict_sequence = predict_sequence.append(accelerator.gather_for_metrics(rna_sequence))
                predict_sequence.append(rna_sequence)
                nonpad_idx.append(rna_value > 0)
            
            predict_express = torch.cat(predict_express, dim = 0)    
            predict_sequence = torch.cat(predict_sequence, dim = 0)
            nonpad_idx = torch.cat(nonpad_idx, dim = 0)
            # accelerator.wait_for_everyone()  

            # Trans back to full length
            # if accelerator.is_main_process:
            predict_express = predict_express.cpu().numpy()
            rna_index_new = predict_sequence.cpu().numpy()
            nonpad_idx = nonpad_idx.cpu().numpy()

            n_cells= len(dataset)
            n_genes = config["model"].get("total_gene")

            rna_index_new -= 1 # for special token <PAD>
            # nonpad_idx = rna_index_new >= 0

            full_zero = np.zeros((n_cells, n_genes))
            full_zero.shape
            predict_matrix = full_zero.copy()

            for cell in tqdm.tqdm(range(n_cells), desc="generating expression matrix", ncols=80, bar_format='{l_bar}{bar}| {percentage:.0f}%'):
                # nonpad_idx = rna_index_new[cell] >= 0
                predict_matrix[cell, rna_index_new[cell][nonpad_idx[cell]]] = predict_express[cell][nonpad_idx[cell]]
                
            predict_express = csr_matrix(predict_matrix, dtype=np.float32)
            predict_rna = sc.AnnData(predict_express, obs=obs)
            predict_rna.var_names = gene['gene_name'].values
            output_rna.append(predict_rna)
            
        output_rna = sc.concat(output_rna)
        output_name = os.path.join(output_dir, name + ".h5ad")
        output_rna.write(output_name)
        print(f"Done! Predict file saved at: '{output_name}'")
        
if __name__ == "__main__":
    main()