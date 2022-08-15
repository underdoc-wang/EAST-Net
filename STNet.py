import torch
from torch import nn, Tensor



class Graph_Convol(nn.Module):      # Graph Signal Propagation Rule
    def __init__(self, n_node:int, h_in:int, h_out:int, cheby_k:int, kappa:float, agg_opt:str, use_bias:bool=True):
        super(Graph_Convol, self).__init__()
        self.n_node = n_node
        self.cheby_k = cheby_k
        self.kappa = kappa

        self.agg_opt = agg_opt
        if agg_opt == 'plain':
            self.proj = nn.Linear(in_features=h_in, out_features=h_out, bias=use_bias)
        elif agg_opt == 'mixhop':
            self.proj = nn.Linear(in_features=h_in*cheby_k, out_features=h_out, bias=use_bias)
        elif agg_opt == 'attention':
            self.att_proj = nn.Linear(in_features=h_in*n_node, out_features=1, bias=use_bias)
            self.proj = nn.Linear(in_features=h_in, out_features=h_out, bias=use_bias)

    def forward(self, x:Tensor, G:Tensor):
        '''
        :param x: (batch, n_node, h_in)
        :param G: (n_node, n_node)
        :return: (batch, n_node, h_out)
        '''

        # generate Chebyshev polynomials
        G_set = [torch.eye(self.n_node).to(G.device), G]    # order 0, 1
        for k in range(2, self.cheby_k):
            G_set.append(torch.mm(2*G, G_set[-1]) - G_set[-2])

        h = x
        if self.agg_opt=='plain':
            for k in range(self.cheby_k):
                h = self.kappa*x + (1-self.kappa)*torch.einsum('bnp,nm->bmp', h, G_set[k])
            out = self.proj(h)
        elif self.agg_opt=='mixhop':
            out = []
            for k in range(self.cheby_k):
                h = self.kappa*x + (1-self.kappa)*torch.einsum('bnp,nm->bmp', x, G_set[k])
                out.append(h)
            out = torch.cat(out, dim=-1)
            out = self.proj(out)
        elif self.agg_opt=='attention':
            out = []
            for k in range(self.cheby_k):
                h = self.kappa * x + (1 - self.kappa) * torch.einsum('bnp,nm->bmp', x, G_set[k])
                out.append(h)
            # get attention score
            att_in = torch.stack(out, dim=1).view(x.shape[0], self.cheby_k, -1)       # (batch, k+1, n_node*h_in)
            att_alpha = torch.softmax(self.att_proj(att_in), dim=1)     # (batch, k, 1)
            out = (att_in * att_alpha).sum(dim=1).reshape(x.shape[0], self.n_node, -1)      # (batch, n_node, h_in)
            out = self.proj(out)
        else:
            raise NotImplementedError

        return out



class GCRU_Cell(nn.Module):       # Graph Convlutional Recurrent Unit
    def __init__(self, n_node:int, h_in:int, h_out:int, cheby_k:int, kappa:float=0.05, agg_opt:str='mixhop', use_bias:bool=True):
        super(GCRU_Cell, self).__init__()
        self.n_node = n_node
        self.h_out = h_out
        self.gates = Graph_Convol(n_node, h_in+h_out, h_out*2, cheby_k, kappa, agg_opt, use_bias)
        self.candi = Graph_Convol(n_node, h_in+h_out, h_out, cheby_k, kappa, agg_opt, use_bias)

    def forward(self, x:Tensor, h:Tensor, G:Tensor):
        '''
        :param x: (batch, n_node, h_in)
        :param h: (batch, n_node, h_out)
        :param G: (n_node, n_node)
        :return: (batch, n_node, h_out)
        '''
        assert len(x.shape) == len(h.shape) == 3

        inp = torch.cat([x, h], dim=-1)
        u_r = self.gates(inp, G)
        u, r = torch.split(u_r, self.h_out, dim=-1)
        update, reset = torch.sigmoid(u), torch.sigmoid(r)

        c = torch.cat([x, reset * h], dim=-1)
        c = torch.tanh(self.candi(c, G))

        return (1.0 - update)*h + update*c

    def init_hidden(self, batch_size:int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(batch_size, self.n_node, self.h_out))
        return hidden



class GCRN_Encoder(nn.Module):        # GCRN Encoder
    def __init__(self, n_node:int, h_in:int, h_out:int or list, cheby_k:int, n_layer:int, use_bias:bool=True):
        super(GCRN_Encoder, self).__init__()
        
        self.n_layer = n_layer
        if not isinstance(h_out, list):
            self.h_dim = self._extend_for_multilayers(h_out)
        else:
            assert len(h_out) == self.n_layer
            self.h_dim = h_out
        
        self.cell_list = nn.ModuleList()
        for i in range(self.n_layer):
            cur_in_dim = h_in if i == 0 else self.h_dim[i - 1]
            self.cell_list.append(GCRU_Cell(n_node, cur_in_dim, self.h_dim[i], cheby_k, use_bias))

    def forward(self, x_seq:Tensor, G:Tensor, h0:list=None):
        '''
        :param x_seq: (batch, t_in, n_node, h_in)
        :param G: (n_node, n_node)
        :param h0: [(batch, n_node, h_out*(2**l))]*n_layer
        :return: h_seq: [(batch, t_in, n_node, h_out*(2**l))]*n_layer, ht: [(batch, n_node, h_out*(2**l))]*n_layer
        '''
        assert len(x_seq.shape) == 4
        batch_size, seq_len, _, _ = x_seq.shape
        if h0 is None:
            h0 = self._init_hidden(batch_size)      # initialize hiddens with zero

        out_seq_list = list()    # layerwise output seq
        ht_list = list()        # layerwise last state
        in_seq_l = x_seq        # current input seq

        for l in range(self.n_layer):
            ht = h0[l]
            out_seq_l = list()
            for t in range(seq_len):
                ht = self.cell_list[l](x=in_seq_l[:,t,:,:], h=ht, G=G)
                out_seq_l.append(ht)

            out_seq_l = torch.stack(out_seq_l, dim=1)   # (batch, t_in, n_node, h_out)
            in_seq_l = out_seq_l
            out_seq_list.append(out_seq_l)
            ht_list.append(ht)

        return out_seq_list, ht_list

    def _init_hidden(self, batch_size:int):
        h0 = []
        for i in range(self.n_layer):
            h0.append(self.cell_list[i].init_hidden(batch_size))
        return h0

    def _extend_for_multilayers(self, h:int):
        h_list = [h] * self.n_layer
        return h_list



class GCRN_Decoder(nn.Module):        # GCRN Decoder
    def __init__(self, n_node:int, h_in:int, h_out:int, cheby_k:int, n_layer:int, use_bias:bool=True):
        super(GCRN_Decoder, self).__init__()
        
        self.n_layer = n_layer
        if not isinstance(h_out, list):
            self.h_dim = self._extend_for_multilayers(h_out)
        else:
            assert len(h_out) == self.n_layer
            self.h_dim = h_out
        
        self.cell_list = nn.ModuleList()
        for i in range(self.n_layer):
            cur_in_dim = h_in if i == 0 else self.h_dim[i-1]
            self.cell_list.append(GCRU_Cell(n_node, cur_in_dim, self.h_dim[i], cheby_k, use_bias))

    def forward(self, xt:Tensor, G:Tensor, ht:list):
        assert len(xt.shape) == 3

        ht_list = list()  # layerwise hidden state
        x_in_l = xt

        for l in range(self.n_layer):
            ht_l = self.cell_list[l](x=x_in_l, h=ht[l], G=G)
            ht_list.append(ht_l)
            x_in_l = ht_l      # update input for next layer

        return ht_l, ht_list

    def _extend_for_multilayers(self, h:int):
        h_list = [h] * self.n_layer
        return h_list



class ST_Net(nn.Module):    # GCRN Encoder-Decoder Framework w/ Time Covariates & ST-Memory
    def __init__(self, n_node:int, c_in:int, h_dim:int, cheby_k:int, n_layer:int, horizon:int, device:str, use_bias:bool=True, adp_graph:bool=True,
                 time_cov:bool=True, tcov_in_dim:int=60, tcov_h_dim:int=2, st_memo:bool=True, st_memo_num:int=4, st_memo_dim:int=8):
        super(ST_Net, self).__init__()
        self.horizon = horizon  # output steps
        self.device = device

        # time_cov or st_memo: rectifications
        self.time_cov = time_cov
        if not self.time_cov:
            self.encoder = GCRN_Encoder(n_node, c_in, h_dim, cheby_k, n_layer, use_bias)
            self.decoder = GCRN_Decoder(n_node, c_in, h_dim, cheby_k, n_layer, use_bias)
        else:
            self.encoder = GCRN_Encoder(n_node, c_in+tcov_h_dim, h_dim, cheby_k, n_layer, use_bias)
            self.decoder = GCRN_Decoder(n_node, c_in+tcov_h_dim, h_dim, cheby_k, n_layer, use_bias)
            self.tcov_h_dim = tcov_h_dim
            self.mlp = nn.Sequential(nn.Linear(in_features=tcov_in_dim, out_features=10, bias=use_bias),
                                     nn.Linear(in_features=10, out_features=tcov_h_dim*n_node, bias=use_bias))

        self.st_memo = st_memo
        if not self.st_memo:
            self.proj = nn.Sequential(nn.Linear(in_features=h_dim, out_features=c_in, bias=use_bias), nn.ReLU())
        else:
            self.n_node = n_node
            self.st_memo_num = st_memo_num
            self.st_memo_dim = st_memo_dim
            self.flat_hidden = n_node * h_dim
            self.st_memory = self.construct_st_memory()
            self.proj = nn.Sequential(
                nn.Linear(in_features=h_dim + st_memo_dim, out_features=c_in, bias=use_bias),
                nn.ReLU())

        # adaptive graph
        if adp_graph:
            self.adpG = self.init_adp_graph(n_node, emb_dim=20)


    def init_adp_graph(self, n_node:int, emb_dim:int):
        node_vec1 = nn.Parameter(torch.randn(n_node, emb_dim).to(self.device), requires_grad=True).to(self.device)    # reassign trainable parameters to self cause bug
        nn.init.xavier_normal_(node_vec1)
        node_vec2 = nn.Parameter(torch.randn(emb_dim, n_node).to(self.device), requires_grad=True).to(self.device)
        nn.init.xavier_normal_(node_vec2)

        adpG = torch.softmax(torch.relu(torch.mm(node_vec1, node_vec2)), dim=1)     # soft
        return adpG

    def construct_st_memory(self):
        # Global ST-Memory
        memory_weight = {}
        memory_weight['memory'] = nn.Parameter(torch.randn(self.st_memo_num, self.st_memo_dim), requires_grad=True).to(self.device)
        nn.init.xavier_normal_(memory_weight['memory'])
        memory_weight['Wa'] = nn.Parameter(torch.randn(self.flat_hidden, self.st_memo_dim), requires_grad=True).to(self.device)
        nn.init.xavier_normal_(memory_weight['Wa'])
        memory_weight['fc'] = nn.Parameter(torch.randn(self.st_memo_dim, self.st_memo_dim*self.n_node), requires_grad=True).to(self.device)
        nn.init.xavier_normal_(memory_weight['fc'])
        return memory_weight

    def query_st_memory(self, h_t:torch.Tensor):
        assert len(h_t.shape) == 3, 'Input to query ST-Memory must be a 3D tensor'
        B, N, h = h_t.shape

        h_t = h_t.reshape(B, N*h)
        query = torch.mm(h_t, self.st_memory['Wa'])
        att_score = torch.softmax(torch.mm(query, self.st_memory['memory'].t()), dim=1)
        att_memory = torch.mm(torch.mm(att_score, self.st_memory['memory']), self.st_memory['fc'])

        return att_memory.reshape(B, N, self.st_memo_dim)

    def forward(self, x_seq:Tensor, t_x:Tensor, t_y:Tensor, G:Tensor=None):
        '''
        :param x_seq: (batch, t_in, n_node, c_in)
        :param t_x: (batch, t_in, t_dim)
        :param t_y: (batch, t_out, t_dim)
        :param G: (n_node, n_node)
        :return: (batch, t_out, n_node, c_in)
        '''
        batch_size, seq_len, n_node, c_in = x_seq.shape

        # graph generating
        if G is None:
            G = self.adpG

        if not self.time_cov:
            # encoding
            _, ht_list = self.encoder(x_seq=x_seq, G=G, h0=None)
            # decoding
            go = torch.zeros((batch_size, n_node, c_in), device=x_seq.device)
            out = []
            for t in range(self.horizon):
                ht, ht_list = self.decoder(xt=go, G=G, ht=ht_list)
                if not self.st_memo:
                    out.append(self.proj(ht))
                else:
                    memory = self.query_st_memory(ht_list[-1])
                    out.append(self.proj(torch.cat([ht, memory], dim=-1)))
                go = out[-1]
        else:
            x_time = self.mlp(t_x).reshape(batch_size, seq_len, n_node, self.tcov_h_dim)
            x_seq = torch.cat([x_seq, x_time], dim=-1)
            # encoding
            _, ht_list = self.encoder(x_seq=x_seq, G=G, h0=None)
            # decoding
            go = torch.zeros((batch_size, n_node, c_in), device=x_seq.device)
            out = []
            for t in range(self.horizon):
                y_time = self.mlp(t_y[:, t, :]).reshape(batch_size, n_node, self.tcov_h_dim)
                go = torch.cat([go, y_time], dim=-1)

                ht, ht_list = self.decoder(xt=go, G=G, ht=ht_list)
                if not self.st_memo:
                    out.append(self.proj(ht))
                else:
                    memory = self.query_st_memory(ht_list[-1])
                    out.append(self.proj(torch.cat([ht, memory], dim=-1)))
                go = out[-1]

        out = torch.stack(out, dim=1)
        return torch.relu(out)


