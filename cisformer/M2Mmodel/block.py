import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from cisformer.M2Mmodel.utils import New_Accelerator as Accelerator

from functools import partial
from flash_attn import flash_attn_func

from distutils.version import LooseVersion
from packaging import version

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')
assert not (
            version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

# helpers

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

# token shifting helper and classes

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# classes

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class PreScaleNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)

class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)
        
        std = 0.1

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        torch.nn.init.xavier_normal_(self.w1.weight, std)
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)
        torch.nn.init.xavier_normal_(self.w2.weight, std)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        qkv_bias = False,
        attn_out_bias = True
    ):
        super().__init__()
        
        std = 0.1
        self.causal = causal
        
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.heads = heads
        
        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        torch.nn.init.xavier_normal_(self.to_q.weight, std)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        torch.nn.init.xavier_normal_(self.to_k.weight, std)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        torch.nn.init.xavier_normal_(self.to_v.weight, std)
        self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
        torch.nn.init.xavier_normal_(self.to_out.weight, std)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, **kwargs):
        h = self.heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        attn_outs = []

        if exists(context_mask):
            global_mask = context_mask[:, None, :, None]
            v.masked_fill_(~global_mask, 0.)
            # print("mask!")

        if exists(pos_emb) and not cross_attend:
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        dtype = q.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            q = q.half()
            k = k.half()
            v = v.half()
            out = flash_attn_func(q.transpose(1,-2), k.transpose(1,-2), v.transpose(1,-2), causal = self.causal)  # 使用flasn_attentions
            out = out.to(dtype)
        else:
            out = flash_attn_func(q.transpose(1,-2), k.transpose(1,-2), v.transpose(1,-2), causal = self.causal)  # 使用flasn_attentions
        attn_outs.append(out.transpose(1,-2))

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        
        return self.dropout(out)

class SelfAttention(Attention):
    def forward(self, *args, context = None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)

class CrossAttention(Attention):
    def forward(self, *args, context = None, **kwargs):
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context = context, **kwargs)


# rotary positional embedding helpers

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


# runner

# for routing arguments into the functions of the reversible layer
def route_args(router, args, depth, cross_attend, only_cross_attn):
    # no decoder self-attention
    if cross_attend and not only_cross_attn:
        routed_args = [(dict(), dict(), dict()) for _ in range(depth)] if cross_attend else [(dict(), dict()) for _ in range(depth)]
    else:
        routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        # no decoder self-attention
        if cross_attend and not only_cross_attn:
            for depth, ((f_args, g_args, h_args), routes) in enumerate(zip(routed_args, router[key])):
                new_f_args, new_g_args, new_h_args = map(lambda route: ({key: val} if route else {}), routes)
                routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args}, {**h_args, **new_h_args})
        else:
            for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
                new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
                routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args

class SequentialSequence(nn.Module):
    def __init__(self, layers, cross_attend, only_cross_attn, args_route = {}):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.cross_attend = cross_attend
        self.layers = layers
        self.args_route = args_route
        self.only_cross_attn = only_cross_attn
        # self.norm = nn.LayerNorm(dim)
        # self.accelerator = Accelerator()

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers), self.cross_attend, self.only_cross_attn)
        layers_and_args = list(zip(self.layers, args))
        
        # no decoder self-attention        
        if self.cross_attend and not self.only_cross_attn:
            for i, ((f, g, h), (f_args, g_args, h_args)) in enumerate(layers_and_args):
                x = x + f(x, **f_args)
                # self.accelerator.print(f"layer{i} ateention mean: {x.mean()} var: {x.var()}")
                # self.accelerator.print(f"layer{i} ateention gpu allocate: {torch.cuda.memory_allocated()/1024**3} g")
                x = x + g(x, **g_args)
                # self.accelerator.print(f"layer{i} cross attention mean: {x.mean()} var: {x.var()}")
                # self.accelerator.print(f"layer{i} cross attention gpu allocate: {torch.cuda.memory_allocated()/1024**3} g")
                x = x + h(x, **h_args)
                # self.accelerator.print(f"layer{i} forward mean: {x.mean()} var: {x.var()}")
                # self.accelerator.print(f"layer{i} forward gpu allocate: {torch.cuda.memory_allocated()/1024**3} g")
        else:
            for i, ((f, g), (f_args, g_args)) in enumerate(layers_and_args):
                x = x + f(x, **f_args)
                # self.accelerator.print(f"layer{i} ateention mean: {x.mean()} var: {x.var()}")
                # self.accelerator.print(f"layer{i} ateention gpu allocate: {torch.cuda.memory_allocated()/1024**3} g")
                x = x + g(x, **g_args)
                # self.accelerator.print(f"layer{i} forward mean: {x.mean()} var: {x.var()}")
                # self.accelerator.print(f"layer{i} forward gpu allocate: {torch.cuda.memory_allocated()/1024**3} g")
        return x
    
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        causal = False,
        ff_mult = 4,
        ff_chunks = 1,
        use_scalenorm = False,
        use_rezero = False,
        ff_glu = False,
        ff_dropout = 0.,
        attn_dropout = 0.,
        cross_attend = False,
        qkv_bias = True,
        attn_out_bias = True,
        shift_tokens = False,
        only_cross_attn = False
    ):
        super().__init__()
        layers = nn.ModuleList([])
        self.heads = heads # important for attention matrix generation
        
        self.only_cross_attn = only_cross_attn
        
        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _ in range(depth):

            attn = SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, dropout = attn_dropout, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)
            ff = Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1)

            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))

            attn, ff = map(wrapper_fn, (attn, ff))

            if not cross_attend:
                layers.append(nn.ModuleList([attn, ff]))
            elif only_cross_attn:
                layers.append(nn.ModuleList([
                    wrapper_fn(CrossAttention(dim, causal = causal, heads = heads, dim_head = dim_head, dropout = attn_dropout, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                    ff
                ]))
            else:
                layers.append(nn.ModuleList([attn,
                    wrapper_fn(CrossAttention(dim, causal = causal, heads = heads, dim_head = dim_head, dropout = attn_dropout, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                    ff
                ]))

        if not cross_attend:
            route_attn = ((True, False),) * depth
            route_context = ((True, False),) * depth
        elif only_cross_attn:
            route_attn = ((True, False),) * depth
            route_context = ((True, False),) * depth
        else:
            route_attn = ((True, True, False),) * depth
            route_context = ((False, True, False),) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = SequentialSequence(layers, cross_attend, only_cross_attn, args_route = {**attn_route_map, **context_route_map})

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)
    
