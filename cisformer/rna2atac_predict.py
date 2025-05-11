import warnings
warnings.filterwarnings("ignore")
import tqdm
# import argparse
import os
import yaml
import torch
import scanpy as sc
from importlib.resources import files as rfiles
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import argparse

# DDP 
# import New_Accelerate as Accelerator
from cisformer.M2Mmodel.utils import * 
from cisformer.M2Mmodel.M2M import M2M_rna2atac #Luz from M2M import M2M_rna2atac
from cisformer.map_rna_atac import is_ens

# 调试工具：梯度异常会对对应位置报错
# torch.autograd.set_detect_anomaly = True
# 设置多线程文件系统
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    
    parser = argparse.ArgumentParser(description='evaluating script of M2M') #Luz parser = argparse.ArgumentParser(description='pretrain script of M2M')
    
    parser.add_argument("-r", "--rna_file", required=True, help="Path of rna adata")
    parser.add_argument("-m", "--model_parameters", required=True, help="Previous trained model parameters")
    parser.add_argument("-o", "--output_dir", default="output", help="Load model")
    parser.add_argument("-n", "--name", default="cisformer_predicted_atac", help="Load model")
    parser.add_argument("-c", "--config_file", default="cisformer_config/rna2atac_config.yaml", help="Config file")
    parser.add_argument("--rna_len", default=3600, help="Load model")
    parser.add_argument("--batch_size", default=2, help="Load model")
    parser.add_argument("--num_workers", default=2, help="Load model")

    # # path
    # parser.add_argument('-d', '--data_dir', type=str, default=None, help="dir of preprocessed paired data")
    # parser.add_argument('-c', '--config_file', type=str, default="config/rna2atac_config.yaml", help="config file for model parameters")
    # parser.add_argument('-l', '--load', type=str, default=None, help="if set, load previous model, path to previous model state_dict")
    # # parser.add_argument('--iconv_weight', type=str, default=None, help='Path ot pretrained iConv weight')
    
    args = parser.parse_args()
    
    # parameters
    # data_dir = data_dir
    rna = sc.read_h5ad(args.rna_file)
    saved_model_path = args.model_parameters
    config_file = args.config_file
    rna_len = int(args.rna_len)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    output_dir = args.output_dir
    name = args.name
    # iconv_weight = args.iconv_weight
    # from config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # init
    SEED = config["training"].get('SEED')
    setup_seed(SEED)
    accelerator = New_Accelerator(step_scheduler_with_optimizer = False)
    
    genes = pd.read_table(rfiles('cisformer.resource')/'human_genes.tsv', names=['gene_ids', 'gene_name'])
    peak_list = pd.read_csv(rfiles("cisformer.resource")/"human_cCREs.bed", sep="\t", header=None)
    peak_list_str = list(peak_list[0]+":"+peak_list[1].map(str)+"-"+peak_list[2].map(str))
    atac_sequence_idx = np.array(range(len(peak_list_str)))
    
    if len(rna.var_names) == len(genes['gene_name']) and all(rna.var_names == genes['gene_name']):
        print("Previous mapped rna detected, skip mapping.")
        rna_new = rna
    else:
        rna.var_names_make_unique()
        rna.X = rna.X.toarray()
        rna_exp = pd.DataFrame(rna.X.T, index=rna.var.index)
        rna_exp.index = rna.var.index
        if is_ens(rna.var_names):
            genes.index = genes['gene_ids'].values
            X_new = pd.merge(genes, rna_exp, how='left', left_index=True, right_index=True).iloc[:, genes.shape[1]:].T
        else:
            genes.index = genes['gene_name'].values
            X_new = pd.merge(genes, rna_exp, how='left', left_index=True, right_index=True).iloc[:, genes.shape[1]:].T
        X_new.fillna(value=0, inplace=True)
        # rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
        # X_new = pd.merge(genes, rna_exp, how='left', left_on='gene_ids').iloc[:, 5:].T
        # X_new.fillna(value=0, inplace=True)
        rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'gene_name': genes['gene_name'], 'feature_types': 'Gene Expression'}))
        rna_new.var.index = genes['gene_name'].values
        rna_new.X = csr_matrix(rna_new.X)
        
    #if os.path.isdir(data_dir):
    #    files = load_data_path(data_dir, greed=True)
    #else:
    #    files = [data_dir]
    
    # lr = float(config["training"].get('lr'))
    # gamma_step = config["training"].get('gamma_step')
    # gamma = config["training"].get('gamma')
    # num_workers = config["training"].get('num_workers')

    # epoch = config["training"].get('epoch')
    # log_every = config["training"].get('log_every')

    # parameters for dataloader construction
    dataloader_kwargs = {'batch_size': batch_size, 'shuffle': False}#Luz True}
    cuda_kwargs = {
        # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
        "num_workers": num_workers,
        'prefetch_factor': 2
    }  
    dataloader_kwargs.update(cuda_kwargs)
    
    with accelerator.main_process_first():
        
        # model loading
        model = M2M_rna2atac(
            dim = config["model"].get("dim"),
            enc_num_gene_tokens = config["model"].get("total_gene") + 1,
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +special tokens
            enc_depth = 0,
            enc_heads = config["model"].get("dec_heads"),
            enc_ff_mult = config["model"].get("dec_ff_mult"),
            enc_dim_head = config["model"].get("dec_dim_head"),
            enc_emb_dropout = config["model"].get("dec_emb_dropout"),
            enc_ff_dropout = config["model"].get("dec_ff_dropout"),
            enc_attn_dropout = config["model"].get("dec_attn_dropout"),
            
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
        if saved_model_path and saved_model_path != "None":
            model.load_state_dict(torch.load(saved_model_path), strict = False)
            accelerator.print("previous model loaded!")
        else:
            accelerator.print("warning: you haven't load previous model!")
        
        # if iconv_weight:
        #     iconv_state_dict = torch.load(iconv_weight)
        #     emb_dict = {"weight": iconv_state_dict["enc.weight"]}
        #     dec_dict = {"weight": iconv_state_dict["dec.weight"], "bias": iconv_state_dict["dec.bias"]}
        #     model.enc.iConv_enc.load_state_dict(emb_dict)
        #     model.dec.iConv_dec.load_state_dict(dec_dict)
        #     accelerator.print("iConv weights loaded!")
            
        #     # iConv freeze
        #     for name, param in model.named_parameters():
        #         if "enc.iConv_enc" in name and iconv_weight:
        #             param.requires_grad = False
        #             accelerator.print(f"{name} weights freezed!")
        #         elif "dec.iConv_dec" in name and iconv_weight:
        #             param.requires_grad = False
        #             accelerator.print(f"{name} weights freezed!")
            
    # #---  Prepare Optimizer ---#
    # optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    # #---  Prepare Scheduler ---#
    # scheduler = StepLR(optimizer, step_size=gamma_step, gamma=gamma) 
    # device = accelerator.device
    model = accelerator.prepare(model)
    #Luz  model, optimizer, train_loader, scheduler = accelerator.prepare(
    #Luz      model, optimizer, train_loader, scheduler
    #Luz  )
    
    # start_date = str(datetime.datetime.now().date())
    #Luz writer = SummaryWriter(os.path.join("logs", f"{start_date}_{save_name}_logs"))
    # global_iter = 0
    #Luz min_loss = 1e6
    
    model.eval()
    # y_hat_out = torch.Tensor() # Luz
    output = []
    with torch.no_grad():
        # files_iter = tqdm.tqdm(files, desc=f"Predicting", ncols=80, total=len(files)) if accelerator.is_main_process else files
        test_dataset = Rna2atacDataset(rna_new, rna_len, atac_sequence_idx, max_express=config["model"]['max_express'])
        test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)
        test_loader = accelerator.prepare(test_loader)
        data_iter = tqdm.tqdm(test_loader, desc=f"Predicting", ncols=80, total=len(test_loader)) if accelerator.is_main_process else test_loader
        
        for rna_sequence, rna_value, atac_sequence, enc_pad_mask in data_iter:
            # rna_value: b, n
            # rna_sequence: b, n, 6
            # rna_pad_mask: b, n
            # atac_sequence: b, n, 6
            # tgt: b, n (A 01 sequence)
                
            #y_hat_out_tmp = torch.Tensor()
            #for j in [0]:#for i in range(atac_sequence.shape[1]//10000+1):
            #    if j != atac_sequence.shape[1]//10000:
            #        atac_sequence_tmp = atac_sequence[:, (j*10000):((j+1)*10000), :]
            #    else:
            #        atac_sequence_tmp = atac_sequence[:, (j*10000):, :]
            #    start = time.time()
                
            # run model
            logist = model(rna_sequence, atac_sequence, rna_value, enc_mask = enc_pad_mask) # b, n
            logist_hat = torch.sigmoid(logist)
            y_hat_out = accelerator.gather_for_metrics(logist_hat).to('cpu')
            if accelerator.is_main_process:
                y_hat_out = (y_hat_out >= 0.5).int()
                y_hat_out = csr_matrix(y_hat_out)
                output.append(y_hat_out)
            # global_iter += 1
            accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output = vstack(output)
        output = sc.AnnData(output, obs=rna_new.obs)
        output.var_names = peak_list_str
        output_name = os.path.join(output_dir, name + ".h5ad")
        os.makedirs(output_dir, exist_ok=True)
        output.write(output_name)
        print(f"Done! Predict file saved at: '{output_name}'")
                
            #Luz if accelerator.is_main_process:    
                #Luz writer.add_scalars("Loss", {"train": epoch_loss / val_num}, e+1)
                #Luz writer.add_scalars("AUROC", {"train": epoch_auroc / val_num}, e+1)
                #Luz writer.add_scalars("AUPRC", {"train": epoch_auprc / val_num}, e+1)
                
                # # 记录梯度
                # for name, param in model.named_parameters():
                #     if param.requires_grad and 'weight' in name:
                #         writer.add_histogram(f"{name}_grad", param.grad, e+1)

            #Luz scheduler.step()
                
            #Luz accelerator.save_model(model, os.path.join(save_path, f"{start_date}_{save_name}"))
            
                     
    # #Luz writer.close()
    # accelerator.print(
    #     "#############################",
    #     "#### Predicting success! ####",
    #     "#############################",
    #     sep = "\n"
    #     )

if __name__ == "__main__":
    main()
