import re
import torch
from torch import nn
from cisformer.M2Mmodel.block import TransformerBlock, default, route_args
from einops import rearrange
import yaml
import torch.nn.functional as F
from torch.cuda.amp import autocast
import threading


ENC_PREFIX = 'enc_'
DEC_PREFIX = 'dec_'

"""
Special tokens:

0: <PAD>
"""

# helper

def exists(val):
    return val is not None

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def input_with_timeout(prompt, timeout):
    """
    Prompt for input with a timeout.

    Args:
        prompt (str): The input prompt message.
        timeout (int): The timeout in seconds.

    Returns:
        str: The input from the user or None if timed out.
    """
    user_input = [None]  # Use a list to store input since lists are mutable

    def get_input():
        user_input[0] = input(prompt)

    # Create and start a thread to get user input
    input_thread = threading.Thread(target=get_input)
    input_thread.start()

    # Wait for the input thread to finish within the timeout period
    input_thread.join(timeout)

    # If the thread is still active, it means we have timed out
    if input_thread.is_alive():
        return None

    return user_input[0]


class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

class Pass(nn.Module):
    def forward(self, x):
        return x

# sinusoidal positional embeddings
# mean 0 var 1

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        inv_freq = 1. / (1000 ** (torch.arange(0, dim, 2).half() / dim))
        position = torch.arange(0, seq_len, dtype=torch.float16)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1).half()
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x).half()
    
# parameter helper    

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return bool(re.match(f'^{prefix}', str))

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def extract_enc_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, dec_kwargs, kwargs

def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if 'mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['mask'])
    return enc_kwargs, dec_kwargs, kwargs


# attntion matrix generator
def whole_attn_score_matrix_generator(q, k, q_mask = None, k_mask = None, double = False):
    # q,k shape: b h n d
    # mask shape: b n
    b, _, n1, _ = q.shape
    _, _, n2, _ = k.shape
    # if n1 == n2:
    #     q = k.clone()
    k_mask = default(k_mask, q_mask)
    output = []
    for batch in range(b):
        if double:
            batch_q = q[batch].double() # h n1 d
            batch_k = k[batch].double() # h n2 d
            batch_attn = torch.matmul(batch_q, batch_k.transpose(-2, -1)) / torch.sqrt(torch.tensor(batch_q.size(-1), dtype=torch.float64)) # h n1 n2
        else:
            batch_q = q[batch].float() # h n1 d
            batch_k = k[batch].float() # h n2 d
            batch_attn = torch.matmul(batch_q, batch_k.transpose(-2, -1)) / torch.sqrt(torch.tensor(batch_q.size(-1), dtype=torch.float32)) # h n1 n2
        # print("batch_attn shape:", batch_attn.shape)
        batch_attn = F.softmax(batch_attn, dim=-1)
        # print("batch_attn shape:", batch_attn.shape)
        batch_attn = torch.mean(batch_attn, dim = 0) # n1, n2
        # print("batch_attn shape:", batch_attn.shape)
        if exists(q_mask):
            batch_q_mask = q_mask[batch] # n1
            # print("batch_q_mask shape:", batch_q_mask.shape)
            batch_attn = batch_attn[batch_q_mask, :]
        if exists(k_mask):
            batch_k_mask = k_mask[batch] # n2
            batch_attn = batch_attn[:, batch_k_mask]
        # if n1 == n2:
        #     batch_attn = (batch_attn + batch_attn.T) / 2
        output.append(batch_attn.to("cpu"))
    return output

def whole_attn_score_matrix(q, k, q_mask = None, k_mask = None, double = False):
    """_summary_

    Args:
        q (_type_): _description_
        k (_type_): _description_
        q_mask (tensor, optional): bool dtype, True for tokens to keep, False to drop. Defaults to None.
        k_mask (tensor, optional): bool dtype, True for tokens to keep, False to drop. Defaults to None.

    Returns:
        list: attention weights for cells
    """
    try:
        with autocast():
            output = whole_attn_score_matrix_generator(q, k, q_mask = q_mask, k_mask = k_mask, double = double)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory, automatically use CPU to compute ...")
            q = q.float().cpu()
            k = k.float().cpu()
            if exists(q_mask):
                q_mask = q_mask.cpu()
            if exists(k_mask):
                k_mask = k_mask.cpu()
            output = whole_attn_score_matrix_generator(q, k, q_mask = q_mask, k_mask = k_mask, double = double)
        else:
            raise e
    return output

# def neighbour_attn_score_matrix(q, k, start=0, end=-1, mask = None):
#     """_summary_

#     Args:
#         q (_type_): _description_
#         k (_type_): _description_
#         start (int, optional): _description_. Defaults to 0.
#         end (int, optional): -1 for whole sequence. Defaults to -1.
#         mask (tensor, optional): bool type, False for pad. Defaults to None.

#     Returns:
#         _type_: _description_
#     """
#     # q,k shape: b h n d
#     # mask shape: b n
#     b, _, n1, _ = q.shape
#     _, _, n2, _ = k.shape
#     output = []
#     for batch in range(b):
#         if exists(mask):
#             batch_mask = mask[batch] # n
#             batch_q = q[batch][:, batch_mask, :] # h n d
#             batch_k = k[batch][:, batch_mask, :] # h n d
#         else:
#             batch_q = q[batch] # h n d
#             batch_k = k[batch] # h n d
#         batch_end = batch_q.shape[1] if end == -1 else end
#         batch_q, batch_k = batch_q[:, start:batch_end, :], batch_k[:, start:batch_end, :] 
#         batch_attn = batch_q @ batch_k.transpose(-1,-2) # h n1 n2
#         batch_attn = torch.mean(batch_attn, dim = 1)
#         # batch_attn = torch.softmax(batch_attn)
#         output.append({"attn": batch_attn.to("cpu"), "q_indices": list(range(start, batch_end)), "k_indices": list(range(start, batch_end))})
#     return output

# def attn_score_matrix(q, k, topest_k = 20, related_num = 5, k_base = True, loop_size = 100):
#     # q,k shape: b h n d
#     if not k_base:
#         q, k = k, q
#     b, h, n1, d = q.shape
#     n2 = k.shape[2]
    
#     # loop 1: find most important k
#     i = 0
#     k_weight = torch.zeros(b,h,n2).to(q.device)
#     # total_s = 0
#     while i < n1:
#         slice_q = q[:, :, i:i+loop_size, :]
#         slice_attn = slice_q @ k.transpose(-1,-2) # b h n1 n2
#         sum_slice_attn = torch.sum(slice_attn, dim = -2)
#         k_weight += sum_slice_attn
#         i += loop_size
#         # total_s += torch.exp(slice_attn).sum().item()
#     k_weight = torch.mean(k_weight, dim = -2) # b, n2
#     _, k_indices = torch.topk(k_weight, topest_k, dim = -1)
    
#     # loop 2: find the most important one related with each topest_k
#     slice_k = []
#     for j in range(k_indices.shape[0]):
#         slice_k.append(k[j, :, k_indices[j, :], :].unsqueeze(0))
#     slice_k = torch.cat(slice_k, dim = 0) # b h topest_k d
#     inner_attn = q @ slice_k.transpose(-1,-2) # b h n1 topest_k
#     inner_attn = torch.mean(inner_attn, dim = 1) # b n1 topest_k
#     inner_q_indices = []
#     for h in range(inner_attn.shape[-1]):
#         slice_inner_attn = inner_attn[:,:,h] # b n1
#         _, slice_inner_q_indices = torch.topk(slice_inner_attn, related_num, dim = -1) # b, topest_q
#         inner_q_indices.append(slice_inner_q_indices)
#     inner_q_indices = torch.cat(inner_q_indices, dim = -1)
#     print(inner_q_indices)
#     q_indices = []
#     attn = []
#     for j in range(inner_q_indices.shape[0]):
#         slice_q_indices = torch.unique(inner_q_indices[j])
#         # print(slice_q_indices.shape)
#         attn.append(inner_attn[j, slice_q_indices, :].unsqueeze(0))
#         # attn.append(torch.exp(inner_attn[j, slice_q_indices, :]) / total_s)
#         q_indices.append(slice_q_indices)
#     attn = torch.cat(attn, dim = 0)
    
#     if k_base:
#         return {"attn": attn.to("cpu"), "q_indices": q_indices, "k_indices": [k_indices[j, :] for j in range(b)]}
#     else:
#         return {"attn": attn.to("cpu"), "k_indices": q_indices, "q_indices": [k_indices[j, :] for j in range(b)]}


# def attn_score_matrix(q, k, start = 0, end = 100, related_num = 5, k_base = True):
#     # q,k shape: b h n d
#     if not k_base:
#         q, k = k, q
#     b, h, n1, d = q.shape
#     k = k[:, :, start:end, :]
#     attn = q @ k.transpose(-1,-2) # b, h, n1, n2
#     attn = torch.mean(attn, dim = 1) # b, n1, n2
    
#     # find most important q
#     _, topest_q_indices = torch.topk(attn, related_num, dim = -2) # b, top_q, n2
#     topest_q_indices = torch.unique(topest_q_indices) # top_q
#     attn = attn[:, topest_q_indices, :] # b, top_q, n2
#     q_indices = topest_q_indices.cpu().tolist()
    
#     if k_base:
#         return {"attn": attn.to("cpu"), "q_indices": q_indices, "k_indices": list(range(start, end))}
#     else:
#        return {"attn": attn.to("cpu"), "k_indices": q_indices, "q_indices": list(range(start, end))}

# module            
class RnaLM(nn.Module):
    """for ATAC decoding

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        use_iConv = False,
        num_gene_tokens = None,
        value_require = False,
        num_value_tokens = None,
        enc = False,
        dim_head = 64,
        causal = False,
        ff_mult = 4,
        ff_chunks = 1,
        use_scalenorm = False,
        use_rezero = False,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        cross_attend = False,
        qkv_bias = True,
        attn_out_bias = True,
        shift_tokens = False,
        only_cross_attn = False
    ):
        super().__init__()
        std = 0.1
        # assert dim % 7 == 0, "dim must be a multiple of 7"

        self.enc = enc
        self.value_require = value_require
        self.use_iConv = use_iConv
        if use_iConv:
            self.iConv_enc = nn.Embedding(10, dim//7)
            torch.nn.init.xavier_normal_(self.iConv_enc.weight, std)
        else:
            assert exists(num_gene_tokens), "you need to pass num_gene_tokens to the model as don't use iConv module"
            self.gene_emb = nn.Embedding(num_gene_tokens, dim)
        if value_require:
            assert exists(num_value_tokens), "you need to pass num_value_tokens to the model"
            self.value_emb = nn.Embedding(num_value_tokens, dim)
            torch.nn.init.xavier_normal_(self.value_emb.weight, std) 
        if not enc:
            assert exists(num_value_tokens), "you need to pass num_value_tokens to the model"
            self.to_out = nn.Linear(dim, num_value_tokens) 
            torch.nn.init.xavier_normal_(self.to_out.weight, std)

        self.pos_emb = Always(0)
        self.layer_pos_emb = Always(None)
        # self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.block = TransformerBlock(dim, depth, heads, dim_head, causal = causal, ff_mult = ff_mult, ff_chunks=ff_chunks, use_scalenorm=use_scalenorm, use_rezero=use_rezero, ff_glu=ff_glu, ff_dropout=ff_dropout, attn_dropout=attn_dropout, cross_attend=cross_attend, qkv_bias=qkv_bias, attn_out_bias=attn_out_bias, shift_tokens=shift_tokens, only_cross_attn = only_cross_attn)
        self.norm = nn.LayerNorm(dim)
        

    def forward(self, x, value = None, **kwargs):       
        # input x shape: batch, n, c
        # input value shape: batch, n
        # token and positional embeddings
        if self.use_iConv:
            b, n, _ = x.shape
            x = self.iConv_enc(x) # b, n, c, d/c
            x = x.view(b, n, -1) # b, n, d
        else:
            b, n = x.shape
            x = self.gene_emb(x) # b, n, d
        if self.value_require:
            x += self.value_emb(value)
        x += self.pos_emb(x)

        x = self.dropout(x)
        layer_pos_emb = self.layer_pos_emb(x)
    
        x = self.block(x, pos_emb = layer_pos_emb, **kwargs)
        
        # norm and to logist
        x = self.norm(x)
        
        if not self.enc:
            x = self.to_out(x)        
        
        return x

class AtacLM(nn.Module):
    """for ATAC decoding

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        enc = False,
        value_require = False,
        dim_head = 64,
        causal = False,
        ff_mult = 4,
        ff_chunks = 1,
        use_scalenorm = False,
        use_rezero = False,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        cross_attend = False,
        qkv_bias = True,
        attn_out_bias = True,
        shift_tokens = False,
        only_cross_attn = False
    ):
        super().__init__()
        std = 0.1
        # assert dim % 7 == 0, "dim must be a multiple of 7"

        self.enc = enc
        self.iConv_enc = nn.Embedding(10, dim//7)
        self.value_require = value_require
        torch.nn.init.xavier_normal_(self.iConv_enc.weight, std)
        if not enc:
            self.to_out = nn.Linear(dim, 1)
            torch.nn.init.xavier_normal_(self.to_out.weight, std)
        if value_require:
            self.value_emb = nn.Embedding(2, dim)
            torch.nn.init.xavier_normal_(self.value_emb.weight, std)

        self.pos_emb = Always(0)
        self.layer_pos_emb = Always(None)
        # self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)
        self.block = TransformerBlock(dim, depth, heads, dim_head, causal = causal, ff_mult = ff_mult, ff_chunks=ff_chunks, use_scalenorm=use_scalenorm, use_rezero=use_rezero, ff_glu=ff_glu, ff_dropout=ff_dropout, attn_dropout=attn_dropout, cross_attend=cross_attend, qkv_bias=qkv_bias, attn_out_bias=attn_out_bias, shift_tokens=shift_tokens, only_cross_attn=only_cross_attn)
        self.norm = nn.LayerNorm(dim)
        

    def forward(self, x, value = None, **kwargs):        
        # token and positional embeddings
        b, n, _ = x.shape
        x = self.iConv_enc(x) # b, n, c, d/c
        x = x.view(b, n, -1) # b, n, d
        if self.value_require:
            x += self.value_emb(value)
        x += self.pos_emb(x)

        x = self.dropout(x)
        layer_pos_emb = self.layer_pos_emb(x)   
        
        x = self.block(x, pos_emb = layer_pos_emb, **kwargs)
        # norm and to logist
        x = self.norm(x)

        if not self.enc:
            x = self.to_out(x).squeeze()

        return x
   

class M2M_rna2atac(nn.Module):
    def __init__(
        self,
        dim,
        **kwargs
    ):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the same dim for both encoder and decoder by passing dim param'

        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['enc'] = True
        enc_kwargs['value_require'] = True
        dec_kwargs['cross_attend'] = True
        dec_kwargs['only_cross_attn'] = True
        
        self.enc = RnaLM(**enc_kwargs)
        self.dec = AtacLM(**dec_kwargs)


    def forward(self, seq_in, seq_out, value, enc_mask = None, dec_mask = None, **kwargs):
        """
        注意参数
        enc_mask
        dec_mask
        """
        
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        enc_kwargs["mask"] = enc_mask
        dec_kwargs['mask'] = dec_mask
        encodings = self.enc(seq_in, value = value, **enc_kwargs)# batch_size, enc_max_seq_len, dim
        
        return self.dec(seq_out, context = encodings, **dec_kwargs)
    
    def generate_attn_weight(self, seq_in, seq_out, value, which = "decoder", enc_mask = None, dec_mask = None, **kwargs):
        """This is used to generate self attention score matrix with the same xlabel and ylabel

        Args:
            seq_in (tensor): input sequence for encoder
            seq_out (tensor): input sequence for decoder
            value (tensor): input value for encoder
            which (str, optional): Which Attention weight you want to generate: "encoder" or "decoder"
            enc_mask (tensor, optional): bool dtype, True for seq_in to keep, False to drop.
            dec_mask (tensor, optional): bool dtype, True for seq_out to keep, False to drop.

        Returns:
            list: attention weights for cells. Cross attention shape: (seq_out, seq_in)
        """
        # if self.value_require:
        #     assert exists(value), "Expression value of rna should be pass as you have set value_require = True"
        assert (which in ["encoder", "decoder"]), 'You can only choose "encoder" or "decoder" for parameter "which"'
        with torch.no_grad():
            enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
            enc_kwargs["mask"] = enc_mask
            dec_kwargs['mask'] = dec_mask
            
            # enc
            if self.enc.use_iConv:
                b, n, _ = seq_in.shape
                x = self.enc.iConv_enc(seq_in)
                x = x.view(b, n, -1)
            else:
                b, n = seq_in.shape
                x = self.enc.gene_emb(seq_in) # b, n, d
            if self.enc.value_require:
                x += self.enc.value_emb(value)
            x += self.enc.pos_emb(x)
            # x = self.enc.dropout(x)
            enc_kwargs['pos_emb'] = self.enc.layer_pos_emb(x)
            
            enc_args = route_args(self.enc.block.net.args_route, enc_kwargs, len(self.enc.block.net.layers), self.enc.block.net.cross_attend, self.enc.block.net.only_cross_attn)
            enc_layers_and_args = list(zip(self.enc.block.net.layers, enc_args))
            
            for i, ((f, g), (f_args, g_args)) in enumerate(enc_layers_and_args):
                if i == len(enc_layers_and_args) - 1 and which == "encoder":
                    q = f.fn.to_q(x)
                    k = f.fn.to_k(x)
                    q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.enc.block.heads), (q, k))
                    enc_attn = whole_attn_score_matrix(q, k, enc_kwargs['mask'], enc_kwargs['mask'])
                    return enc_attn
                x = x + f(x, **f_args)
                x = x + g(x, **g_args)
                
            encodings = self.enc.norm(x)
            
            # dec
            b, n, _ = seq_out.shape
            x = self.dec.iConv_enc(seq_out)
            x = x.view(b, n, -1)
            x += self.dec.pos_emb(x)
            # x = self.dec.dropout(x)
            dec_kwargs['pos_emb'] = self.dec.layer_pos_emb(x)
            dec_kwargs['context'] = encodings
            
            dec_args = route_args(self.dec.block.net.args_route, dec_kwargs, len(self.dec.block.net.layers), self.dec.block.net.cross_attend, self.dec.block.net.only_cross_attn)
            dec_layers_and_args = list(zip(self.dec.block.net.layers, dec_args))
            
            for i, ((f, g), (f_args, g_args)) in enumerate(dec_layers_and_args):
                if i == len(dec_layers_and_args) - 1 and which == "decoder":
                    q = f.fn.to_q(x)
                    k = f.fn.to_k(encodings)
                    q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.dec.block.heads), (q, k))
                    cross_attn = whole_attn_score_matrix(q, k, dec_kwargs['mask'], enc_kwargs['mask'])
                    # cross_attn = neighbour_attn_score_matrix(q, k, start, end)
                    return cross_attn
                x = x + f(x, **f_args)
                x = x + g(x, **g_args)
            
    # def generate_self_attn_score_matrix(self, seq_in, seq_out, value, start=0, end=100, encoder = True, **kwargs):
    #     """This is used to generate self attention score matrix with the same xlabel and ylabel

    #     Args:
    #         seq_in (tensor): input sequence for encoder
    #         seq_out (tensor): input sequence for decoder
    #         value (tensor): expression value of RNA
    #         start (int, optional): Start index for attention matrix generation. Defaults to 0.
    #         end (int, optional): End index for attention matrix generation. Defaults to 100.
    #         encoder (bool, optional): Set True to generate self attention score for encoder else decoder. Defaults to True.
            
    #         You should also send enc_mask and dec_mask !

    #     Returns:
    #         self_attn (dict)
    #     """
    #     with torch.no_grad():
    #         enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
            
    #         # enc
    #         if self.enc.use_iConv:
    #             b, n, _ = seq_in.shape
    #             x = self.enc.iConv_enc(seq_in)
    #             x = x.view(b, n, -1)
    #         else:
    #             b, n = seq_in.shape
    #             x = self.enc.gene_emb(seq_in)
    #         x += self.enc.value_emb(value)
    #         x += self.enc.pos_emb(x)
    #         # x = self.enc.dropout(x)
    #         enc_kwargs['pos_emb'] = self.enc.layer_pos_emb(x)
            
    #         enc_args = route_args(self.enc.block.net.args_route, enc_kwargs, len(self.enc.block.net.layers), self.enc.block.net.cross_attend)
    #         enc_layers_and_args = list(zip(self.enc.block.net.layers, enc_args))
            
    #         for i, ((f, g), (f_args, g_args)) in enumerate(enc_layers_and_args):
    #             if i == len(enc_layers_and_args) - 1 and encoder:
    #                 q = f.fn.to_q(x).float()
    #                 k = f.fn.to_k(x).float()
    #                 q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.enc.block.heads), (q, k))
    #                 enc_attn = neighbour_attn_score_matrix(q, k, start, end)
    #                 return enc_attn
    #             x = x + f(x, **f_args)
    #             x = x + g(x, **g_args)
                
    #         encodings = self.enc.norm(x)
            
    #         # dec
    #         b, n, _ = seq_out.shape
    #         x = self.dec.iConv_enc(seq_out) # b, n, c, d/c
    #         x = x.view(b, n, -1) # b, n, d
    #         x += self.dec.pos_emb(x)
    #         # x = self.dec.dropout(x)
    #         dec_kwargs['pos_emb'] = self.dec.layer_pos_emb(x)
    #         dec_kwargs['context'] = encodings
            
    #         dec_args = route_args(self.dec.block.net.args_route, dec_kwargs, len(self.dec.block.net.layers), self.dec.block.net.cross_attend)
    #         dec_layers_and_args = list(zip(self.dec.block.net.layers, dec_args))
            
    #         for i, ((f, g, h), (f_args, g_args, h_args)) in enumerate(dec_layers_and_args):
    #             if i == len(enc_layers_and_args) - 1:
    #                 q = f.fn.to_q(x).float()
    #                 k = f.fn.to_k(x).float()
    #                 q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.dec.block.heads), (q, k))
    #                 dec_attn = neighbour_attn_score_matrix(q, k, start, end)
    #                 return dec_attn
    #             x = x + f(x, **f_args)
    #             x = x + g(x, **g_args)
    #             x = x + h(x, **h_args)
            
            
    # def generate_cross_attn_score_matrix(self, seq_in, seq_out, value, start = 0, end = 100, related_num = 5, k_base = True, **kwargs):
    #     """This is used to generate cross attention matrix.

    #     Args:
    #         seq_in (tensor): _description_
    #         seq_out (tensor): _description_
    #         value (tensor): expression value of RNA
    #         topest_k (int, optional): How many elements in k should be kept. You can consider k is the sequence in encoder. Defaults to 20.
    #         related_num (int, optional): The number of elements in seq_out which are most related with element in k should be kept. Defaults to 5.
    #         k_base (bool, optional): Whether to exchange the meaning for q and k (seq_in and seq_out). Defaults to True.
    #         loop_size (int, optional): A parameter for faster compute bui will increase memory usage. Defaults to 100.
            
    #         You should also send enc_mask and dec_mask !

    #     Returns:
    #         cross_attn (dict):
    #     """
    #     with torch.no_grad():
    #         enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
            
    #         # enc
    #         if self.enc.use_iConv:
    #             b, n, _ = seq_in.shape
    #             x = self.enc.iConv_enc(seq_in)
    #             x = x.view(b, n, -1)
    #         else:
    #             b, n = seq_in.shape
    #             x = self.enc.gene_emb(seq_in)
    #         x += self.enc.value_emb(value)
    #         x += self.enc.pos_emb(x)
    #         # x = self.enc.dropout(x)
    #         enc_kwargs['pos_emb'] = self.enc.layer_pos_emb(x)
            
    #         enc_args = route_args(self.enc.block.net.args_route, enc_kwargs, len(self.enc.block.net.layers), self.enc.block.net.cross_attend)
    #         enc_layers_and_args = list(zip(self.enc.block.net.layers, enc_args))
            
    #         for i, ((f, g), (f_args, g_args)) in enumerate(enc_layers_and_args):
    #             x = x + f(x, **f_args)
    #             x = x + g(x, **g_args)
                
    #         encodings = self.enc.norm(x)
            
    #         # dec
    #         b, n, _ = seq_out.shape
    #         x = self.dec.iConv_enc(seq_out) # b, n, c, d/c
    #         x = x.view(b, n, -1) # b, n, d
    #         x += self.dec.pos_emb(x)
    #         # x = self.dec.dropout(x)
    #         dec_kwargs['pos_emb'] = self.dec.layer_pos_emb(x)
    #         dec_kwargs['context'] = encodings
            
    #         dec_args = route_args(self.dec.block.net.args_route, dec_kwargs, len(self.dec.block.net.layers), self.dec.block.net.cross_attend)
    #         dec_layers_and_args = list(zip(self.dec.block.net.layers, dec_args))
            
    #         for i, ((f, g, h), (f_args, g_args, h_args)) in enumerate(dec_layers_and_args):
    #             if i == len(enc_layers_and_args) - 1:
    #                 q = g.fn.to_q(x).float()
    #                 k = g.fn.to_k(encodings).float()
    #                 q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.dec.block.heads), (q, k))
    #                 cross_attn = attn_score_matrix(q, k, start = start, end = end, related_num = related_num, k_base = k_base)
    #                 # cross_attn = neighbour_attn_score_matrix(q, k, start, end)
    #                 return cross_attn
    #             x = x + f(x, **f_args)
    #             x = x + g(x, **g_args)
    #             x = x + h(x, **h_args)



class M2M_atac2rna(nn.Module):
    """_summary_

    Model input: seq_in, seq_out, enc_mask
    Attention: When you want to generate attention score matrix, you should also pass "enc_mask"
    """
    def __init__(
        self,
        dim,
        # value_require = False,
        **kwargs
    ):
        super().__init__()
        # self.value_require = value_require
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the same dim for both encoder and decoder by passing dim param'

        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['enc'] = True
        dec_kwargs['cross_attend'] = True
        
        self.enc = AtacLM(**enc_kwargs)
        # self.dec = RnaLM(value_require = value_require, **dec_kwargs)
        self.dec = RnaLM(**dec_kwargs)


    # def forward(self, seq_in, seq_out, value = None, **kwargs): # should also send enc_mask
    def forward(self, seq_in, seq_out, enc_mask = None, dec_mask = None, **kwargs): # should also send enc_mask
        # if self.value_require:
        #     assert exists(value), "Expression value of RNA should be pass as you have set value_require = True"
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        enc_kwargs["mask"] = enc_mask
        dec_kwargs['mask'] = dec_mask
        encodings = self.enc(seq_in, **enc_kwargs)# batch_size, enc_max_seq_len, dim
        
        # return self.dec(seq_out, value = value, context = encodings, **dec_kwargs)
        output = self.dec(seq_out, context = encodings, **dec_kwargs)
        # output[:,:,0] += 10
        return output
        # return encodings
        
    
    # def generate_self_attn_score_matrix(self, seq_in, seq_out, value=None, start=0, end=100, encoder = True, **kwargs):
    def generate_attn_weight(self, seq_in, seq_out, which = "encoder", enc_mask = None, dec_mask = None, **kwargs):
        """This is used to generate self attention score matrix with the same xlabel and ylabel

        Args:
            seq_in (tensor): input sequence for encoder
            seq_out (tensor): input sequence for decoder
            which (str, optional): Which Attention weight you want to generate: "encoder", "decoder" or "cross"
            enc_mask (tensor, optional): bool dtype, True for seq_in to keep, False to drop.
            dec_mask (tensor, optional): bool dtype, True for seq_out to keep, False to drop.

        Returns:
            list: attention weights for cells. Cross attention shape: (seq_out, seq_in)
        """
        # if self.value_require:
        #     assert exists(value), "Expression value of rna should be pass as you have set value_require = True"
        assert (which in ["encoder", "decoder", "cross"]), 'You can only choose "encoder", "decoder" or "cross" for parameter which'
        with torch.no_grad():
            enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
            enc_kwargs["mask"] = enc_mask
            dec_kwargs['mask'] = dec_mask
            
            # enc
            b, n, _ = seq_in.shape
            x = self.enc.iConv_enc(seq_in)
            x = x.view(b, n, -1)
            x += self.enc.pos_emb(x)
            # x = self.enc.dropout(x)
            enc_kwargs['pos_emb'] = self.enc.layer_pos_emb(x)
            
            enc_args = route_args(self.enc.block.net.args_route, enc_kwargs, len(self.enc.block.net.layers), self.enc.block.net.cross_attend, self.enc.block.net.only_cross_attn)
            enc_layers_and_args = list(zip(self.enc.block.net.layers, enc_args))
            
            for i, ((f, g), (f_args, g_args)) in enumerate(enc_layers_and_args):
                if i == len(enc_layers_and_args) - 1 and which == "encoder":
                    q = f.fn.to_q(x)
                    k = f.fn.to_k(x)
                    # print(q)
                    # print(k)
                    q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.enc.block.heads), (q, k))
                    enc_attn = whole_attn_score_matrix(q, k, enc_kwargs['mask'], enc_kwargs['mask'])
                    return enc_attn
                x = x + f(x, **f_args)
                x = x + g(x, **g_args)
                
            encodings = self.enc.norm(x)
            
            # dec
            if self.dec.use_iConv:
                b, n, _ = seq_out.shape
                x = self.dec.iConv_enc(seq_out) # b, n, c, d/c
                x = x.view(b, n, -1) # b, n, d
            else:
                b, n = seq_out.shape
                x = self.dec.gene_emb(seq_out) # b, n, d
            # if self.value_require:
            #     x += self.dec.value_emb(value)
            x += self.dec.pos_emb(x)
            # x = self.dec.dropout(x)
            dec_kwargs['pos_emb'] = self.dec.layer_pos_emb(x)
            dec_kwargs['context'] = encodings
            
            dec_args = route_args(self.dec.block.net.args_route, dec_kwargs, len(self.dec.block.net.layers), self.dec.block.net.cross_attend, self.enc.block.net.only_cross_attn)
            dec_layers_and_args = list(zip(self.dec.block.net.layers, dec_args))
            
            for i, ((f, g, h), (f_args, g_args, h_args)) in enumerate(dec_layers_and_args):
                if i == len(dec_layers_and_args) - 1 and which == "decoder":
                    q = f.fn.to_q(x)
                    k = f.fn.to_k(x)
                    q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.dec.block.heads), (q, k))
                    # dec_attn = neighbour_attn_score_matrix(q, k, start, end)
                    dec_attn = whole_attn_score_matrix(q, k, dec_kwargs['mask'], dec_kwargs['mask'])
                    return dec_attn
                if i == len(dec_layers_and_args) - 1 and which == "cross":
                    q = g.fn.to_q(x)
                    k = g.fn.to_k(encodings)
                    q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.dec.block.heads), (q, k))
                    cross_attn = whole_attn_score_matrix(q, k, dec_kwargs['mask'], enc_kwargs['mask'])
                    # cross_attn = neighbour_attn_score_matrix(q, k, start, end)
                    return cross_attn
                x = x + f(x, **f_args)
                x = x + g(x, **g_args)
                x = x + h(x, **h_args)
            
            
    # # def generate_cross_attn_score_matrix(self, seq_in, seq_out, value = None, start = 0, end = 100, related_num = 5, k_base = True, **kwargs):
    # def generate_cross_attn_score_matrix(self, seq_in, seq_out, **kwargs):
    #     """This is used to generate cross attention matrix.

    #     Args:
    #         seq_in (tensor): _description_
    #         seq_out (tensor): _description_
    #         value (tensor): expression value of RNA
    #         topest_k (int, optional): How many elements in k should be kept. You can consider k is the sequence in encoder. Defaults to 20.
    #         related_num (int, optional): The number of elements in seq_out which are most related with element in k should be kept. Defaults to 5.
    #         k_base (bool, optional): Whether to exchange the meaning for q and k (seq_in and seq_out). Defaults to True.
    #         loop_size (int, optional): A parameter for faster compute bui will increase memory usage. Defaults to 100.
            
    #         You should also send enc_mask and dec_mask !

    #     Returns:
    #         list: Attention weights for cells. Shape (len(seq_in), len(seq_out))
    #     """
    #     # if self.value_require:
    #     #     assert exists(value), "Expression value of rna should be pass as you have set value_require = True"
    #     with torch.no_grad():
    #         enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
            
    #         # enc
    #         b, n, _ = seq_in.shape
    #         x = self.enc.iConv_enc(seq_in)
    #         x = x.view(b, n, -1)
    #         x += self.enc.pos_emb(x)
    #         # x = self.enc.dropout(x)
    #         enc_kwargs['pos_emb'] = self.enc.layer_pos_emb(x)
            
    #         enc_args = route_args(self.enc.block.net.args_route, enc_kwargs, len(self.enc.block.net.layers), self.enc.block.net.cross_attend)
    #         enc_layers_and_args = list(zip(self.enc.block.net.layers, enc_args))
            
    #         for i, ((f, g), (f_args, g_args)) in enumerate(enc_layers_and_args):
    #             x = x + f(x, **f_args)
    #             x = x + g(x, **g_args)
                
    #         encodings = self.enc.norm(x)
            
    #         # dec
    #         if self.dec.use_iConv:
    #             b, n, _ = seq_out.shape
    #             x = self.dec.iConv_enc(seq_out) # b, n, c, d/c
    #             x = x.view(b, n, -1) # b, n, d
    #         else:
    #             b, n = seq_out.shape
    #             x = self.dec.gene_emb(x) # b, n, d
    #         # if self.value_require:
    #         #     x += self.dec.value_emb(value)
    #         x += self.dec.pos_emb(x)
    #         # x = self.dec.dropout(x)
    #         dec_kwargs['pos_emb'] = self.dec.layer_pos_emb(x)
    #         dec_kwargs['context'] = encodings
            
    #         dec_args = route_args(self.dec.block.net.args_route, dec_kwargs, len(self.dec.block.net.layers), self.dec.block.net.cross_attend)
    #         dec_layers_and_args = list(zip(self.dec.block.net.layers, dec_args))
            
    #         for i, ((f, g, h), (f_args, g_args, h_args)) in enumerate(dec_layers_and_args):
    #             if i == len(enc_layers_and_args) - 1:
    #                 q = g.fn.to_q(x)
    #                 k = g.fn.to_k(encodings)
    #                 q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.dec.block.heads), (q, k))
    #                 cross_attn = attn_score_matrix(q, k, start = start, end = end, related_num = related_num, k_base = k_base)
    #                 # cross_attn = neighbour_attn_score_matrix(q, k, start, end)
    #                 return cross_attn
    #             x = x + f(x, **f_args)
    #             x = x + g(x, **g_args)
    #             x = x + h(x, **h_args)