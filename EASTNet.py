import torch
from torch import nn, Tensor



class Graph_Convol(nn.Module):      # Graph Signal Propagation Rule
    def __init__(self, n_node:int, h_in:int, h_out:int, cheby_k:int, kappa:float, agg_opt:str, use_bias:bool=True):
        super(Graph_Convol, self).__init__()
        self.n_node = n_node
        self.cheby_k = cheby_k
        self.kappa = kappa

        self.agg_opt = 'mixhop'
        # if agg_opt == 'plain':
        #     self.proj = nn.Linear(in_features=h_in, out_features=h_out, bias=use_bias)
        # elif agg_opt == 'mixhop':
        #     self.proj = nn.Linear(in_features=h_in*cheby_k, out_features=h_out, bias=use_bias)
        # elif agg_opt == 'attention':
        #     self.att_proj = nn.Linear(in_features=h_in*n_node, out_features=1, bias=use_bias)
        #     self.proj = nn.Linear(in_features=h_in, out_features=h_out, bias=use_bias)

    def forward(self, x:Tensor, G:Tensor, Wgc:Tensor):
        '''
        :param x: (batch, n_node, h_in)
        :param G: (n_node, n_node)
        :param W: (batch, h_in*cheby_k, h_out) - mixhop
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
            #out = self.proj(out)
            out = torch.einsum('bnk,bkq->bnq', out, Wgc)

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
        self.h_in = h_in
        self.h_out = h_out
        self.cheby_k = cheby_k
        self.gates = Graph_Convol(n_node, h_in+h_out, h_out*2, cheby_k, kappa, agg_opt, use_bias)
        self.candi = Graph_Convol(n_node, h_in+h_out, h_out, cheby_k, kappa, agg_opt, use_bias)

    def forward(self, x:Tensor, h:Tensor, G:Tensor, W3:Tensor, batch:int=32):
        '''
        :param x: (batch, n_node, h_in)
        :param h: (batch, n_node, h_out)
        :param G: (n_node, n_node)
        :param W3: (batch, 3, h_in*cheby_k, h_out): for u, r, c
        :return: (batch, n_node, h_out)
        '''
        assert len(x.shape) == len(h.shape) == 3

        inp = torch.cat([x, h], dim=-1)
        # reshape weight
        W3 = W3.reshape(batch, 3*(self.h_in + self.h_out)*self.cheby_k, self.h_out)
        Wu, Wr, Wc = torch.split(W3, (self.h_in + self.h_out)*self.cheby_k, dim=1)

        u_r = self.gates(inp, G, torch.cat([Wu, Wr], dim=-1))
        u, r = torch.split(u_r, self.h_out, dim=-1)
        update, reset = torch.sigmoid(u), torch.sigmoid(r)

        c = torch.cat([x, reset * h], dim=-1)
        c = torch.tanh(self.candi(c, G, Wc))

        return (1.0 - update)*h + update*c

    def init_hidden(self, batch_size:int):
        hidden = torch.zeros(batch_size, self.n_node, self.h_out)
        return hidden



class GCRN_Encoder(nn.Module):      # GCRN Encoder
    def __init__(self, n_node: int, h_in: int, h_out: int or list, cheby_k: int, n_layer: int, device:str,
                 use_bias: bool = True, pyramid: bool = False, factor: int = 2):
        super(GCRN_Encoder, self).__init__()
        self.device = device
        self.pyramid = pyramid      # whether do pyramidal structure (w/ factor of 2)
        self.factor = factor if self.pyramid else 1

        self.n_layer = n_layer
        if not isinstance(h_out, list):
            self.h_dim = self._extend_for_multilayers(h_out)
        else:
            assert len(h_out) == self.n_layer
            self.h_dim = h_out

        self.cell_list = nn.ModuleList()
        for i in range(self.n_layer):
            cur_in_dim = h_in if i == 0 else self.h_dim[i - 1] * (self.factor ** i)
            self.cell_list.append(GCRU_Cell(n_node, cur_in_dim, self.h_dim[i], cheby_k))

    def forward(self, x_seq: Tensor, G: Tensor, W3l:list, h0: list = None):
        '''
        :param x_seq: (batch, t_in, n_node, h_in)
        :param G: (n_node, n_node)
        :param W3l: [(3, h_in*cheby_k, h_out)]*n_layer
        :param h0: [(batch, n_node, h_out*(2**l))]*n_layer
        :return: h_seq: [(batch, t_in, n_node, h_out*(2**l))]*n_layer, ht: [(batch, n_node, h_out*(2**l))]*n_layer
        '''
        assert len(x_seq.shape) == 4
        batch_size, seq_len, _, _ = x_seq.shape
        if h0 is None:
            h0 = self._init_hidden(batch_size)  # initialize hiddens with zero

        out_seq_list = list()  # layerwise output seq
        ht_list = list()  # layerwise last state
        in_seq_l = x_seq  # current input seq

        for l in range(self.n_layer):
            ht = h0[l]
            out_seq_l = list()
            for t in range(seq_len):
                ht = self.cell_list[l](x=in_seq_l[:, t, :, :], h=ht, G=G, W3=W3l[l])
                out_seq_l.append(ht)

            out_seq_l = torch.stack(out_seq_l, dim=1)  # (batch, t_in, n_node, h_out)
            if not self.pyramid:
                in_seq_l = out_seq_l
            else:
                factor_list = []
                for f in range(self.factor):
                    factor_list.append(
                        torch.stack([out_seq_l[:, f + t * self.factor, :, :] for t in range(seq_len // self.factor)], dim=1))
                    in_seq_l = torch.cat(factor_list, dim=-1)
                seq_len //= self.factor

            out_seq_list.append(out_seq_l)
            ht_list.append(ht)

        return out_seq_list, ht_list

    def _init_hidden(self, batch_size: int):
        h0 = []
        for i in range(self.n_layer):
            h0.append(self.cell_list[i].init_hidden(batch_size).to(self.device))
        return h0

    def _extend_for_multilayers(self, h: int):
        h_list = [h] * self.n_layer
        return h_list



class GCRN_Decoder(nn.Module):  # GCRN Decoder
    def __init__(self, n_node: int, h_in: int, h_out: int, cheby_k: int, n_layer: int, use_bias: bool = True,
                 pyramid: bool = False, factor: int = 2):
        super(GCRN_Decoder, self).__init__()
        self.pyramid = pyramid
        self.factor = factor if self.pyramid else 1

        self.n_layer = n_layer
        if not isinstance(h_out, list):
            self.h_dim = self._extend_for_multilayers(h_out)
        else:
            assert len(h_out) == self.n_layer
            self.h_dim = h_out

        self.cell_list = nn.ModuleList()
        for i in range(self.n_layer):
            cur_in_dim = h_in if i == 0 else self.h_dim[i - 1]
            self.cell_list.append(GCRU_Cell(n_node, cur_in_dim, self.h_dim[i], cheby_k))

    def forward(self, xt: Tensor, G: Tensor, W3l:list, ht: list):
        assert len(xt.shape) == 3

        ht_list = list()  # layerwise hidden state
        x_in_l = xt

        for l in range(self.n_layer):
            ht_l = self.cell_list[l](x=x_in_l, h=ht[l], G=G, W3=W3l[l])
            ht_list.append(ht_l)
            x_in_l = ht_l  # update input for next layer

        return ht_l, ht_list

    def _extend_for_multilayers(self, h: int):
        h_list = [h] * self.n_layer
        return h_list



class EAST_Net(nn.Module):      # HMIN Encoder-Decoder Framework w/ MDFG
    def __init__(self, n_node: int, c_in: int, h_dim: int, cheby_k: int, n_layer: int, horizon: int, device: str,
                 use_bias: bool = True, adp_graph: bool = True, pyramid: bool = True, factor: int = 2,
                 time_cov: bool = True, tcov_in_dim: int = 60, tcov_h_dim: int = 2, east_memo_num:int=8, east_memo_dim:int=16):
        super(EAST_Net, self).__init__()
        self.n_layer = n_layer
        self.horizon = horizon  # output steps
        self.device = device
        factor = factor if pyramid else 1

        # adaptive graph
        self.adp_graph = adp_graph
        if self.adp_graph:
            self.sp_emb_vecs = self.init_adp_graph(n_node, emb_dim=20)
            self.se_emb_vecs = self.init_adp_graph(c_in, emb_dim=3)

        # encoder-decoder: spatial & semantic
        self.encoder_sp = GCRN_Encoder(n_node, c_in+tcov_h_dim, h_dim, cheby_k, n_layer, device, use_bias, pyramid=pyramid)
        self.decoder_sp = GCRN_Decoder(n_node, c_in+tcov_h_dim, h_dim, cheby_k, n_layer, use_bias, pyramid=pyramid)

        self.encoder_se = GCRN_Encoder(c_in, n_node+tcov_h_dim, h_dim, cheby_k, n_layer, device, use_bias, pyramid=pyramid)
        self.decoder_se = GCRN_Decoder(c_in, n_node+tcov_h_dim, h_dim, cheby_k, n_layer, use_bias, pyramid=pyramid)

        # time covariates
        self.tcov_emb = nn.Linear(in_features=tcov_in_dim, out_features=8, bias=use_bias)
        self.tcov_lin_sp = nn.Linear(in_features=8, out_features=tcov_h_dim*n_node, bias=use_bias)
        self.tcov_lin_se = nn.Linear(in_features=8, out_features=tcov_h_dim*c_in, bias=use_bias)

        # edge generation output
        self.proj = nn.Linear(in_features=h_dim, out_features=h_dim, bias=False)

        # EAST memory
        self.n_node = n_node
        self.c_in = c_in
        self.h_dim = h_dim
        self.tcov_h_dim = tcov_h_dim
        self.east_memo_num = east_memo_num
        self.east_memo_dim = east_memo_dim
        # initialize
        self.east_memo = self.construct_east_memo()
        # filter generation network
        self.chunk_list = []
        params_num = 0
        for i in range(self.n_layer):
            Wgc = 0
            if i==0:
                Wgc = (c_in+tcov_h_dim+h_dim)*cheby_k*h_dim       # params in layer-1 spatial GCRU encoder
                self.chunk_list.append(Wgc*3)
                params_num += Wgc*3
                Wgc = (c_in+tcov_h_dim+h_dim)*cheby_k*h_dim       # params in layer-1 spatial GCRU decoder
                self.chunk_list.append(Wgc*3)
                params_num += Wgc*3
                Wgc = (n_node+tcov_h_dim+h_dim)*cheby_k*h_dim     # params in layer-1 semantic GCRU encoder
                self.chunk_list.append(Wgc*3)
                params_num += Wgc*3
                Wgc = (n_node+tcov_h_dim+h_dim)*cheby_k*h_dim     # params in layer-1 semantic GCRU decoder
                self.chunk_list.append(Wgc*3)
                params_num += Wgc*3
            else:
                Wgc = h_dim*(factor+1)*cheby_k*h_dim      # params in layer-l spatial GCRU encoder
                self.chunk_list.append(Wgc*3)
                params_num += Wgc*3
                Wgc = h_dim*2*cheby_k*h_dim      # params in layer-l spatial GCRU decoder
                self.chunk_list.append(Wgc*3)
                params_num += Wgc*3
                Wgc = h_dim*(factor+1)*cheby_k*h_dim      # params in layer-l semantic GCRU encoder
                self.chunk_list.append(Wgc*3)
                params_num += Wgc*3
                Wgc = h_dim*2*cheby_k*h_dim      # params in layer-l semantic GCRU decoder
                self.chunk_list.append(Wgc*3)
                params_num += Wgc*3
        self.fgn = nn.Sequential(nn.Linear(in_features=self.east_memo_dim, out_features=64, bias=True),
                                 nn.Linear(in_features=64, out_features=params_num, bias=True))

    def construct_east_memo(self):
        memory_weight = nn.ParameterDict()
        memory_weight['memory'] = nn.Parameter(torch.randn(self.east_memo_num, self.east_memo_dim), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['memory'])
        flat_hidden = (self.n_node+self.c_in)*self.h_dim
        memory_weight['Wa'] = nn.Parameter(torch.randn(flat_hidden, self.east_memo_dim), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['Wa'])
        return memory_weight

    def query_east_memo(self, ht_sp:Tensor, ht_se:Tensor):    # generate parameters based on EAST memory
        ht_mob = torch.cat([ht_sp, ht_se], dim=1)
        ht_flat = ht_mob.view(ht_mob.shape[0], -1)

        # query event prototype
        query = torch.matmul(ht_flat, self.east_memo['Wa'])
        att_score = torch.softmax(torch.matmul(query, self.east_memo['memory'].t()), dim=-1)    # (batch, memo_num)
        att_memo = torch.matmul(att_score, self.east_memo['memory'])    # (batch, memo_dim)

        # generate parameters
        params_flat = self.fgn(att_memo)
        params_list = torch.split(params_flat, self.chunk_list, dim=-1)
        assert len(params_list) == self.n_layer * 4
        params_sp_enc = params_list[0::4]
        params_sp_dec = params_list[1::4]
        params_se_enc = params_list[2::4]
        params_se_dec = params_list[3::4]
        return params_sp_enc, params_se_enc, params_sp_dec, params_se_dec

    def init_adp_graph(self, n_node: int, emb_dim: int):
        emb_vecs = nn.ParameterList()
        emb_vecs.append(nn.Parameter(torch.randn(n_node, emb_dim), requires_grad=True))
        nn.init.xavier_normal_(emb_vecs[-1])
        emb_vecs.append(nn.Parameter(torch.randn(emb_dim, n_node), requires_grad=True))
        nn.init.xavier_normal_(emb_vecs[-1])
        return emb_vecs

    def forward(self, x_seq: Tensor, t_x: Tensor, t_y: Tensor, ht_pair:tuple, G: tuple = None):
        '''
        :param x_seq: (batch, t_in, n_node, c_in)
        :param t_x: (batch, t_in, t_dim)
        :param t_y: (batch, t_out, t_dim)
        :param ht_pair: [(batch, n_node, h_dim), (batch, c_in, h_dim)]
        :param G: [(n_node, n_node), (c_in, c_in)]
        :return: y_hat_seq (batch, t_out, n_node, c_in), ht_pair [(batch, n_node, h_dim), (batch, c_in, h_dim)]
        '''
        assert len(x_seq.shape) == 4
        batch_size, seq_len, n_node, c_in = x_seq.shape
        if (self.adp_graph)&(G is None):
            Gsp = torch.softmax(torch.relu(torch.mm(self.sp_emb_vecs[0], self.sp_emb_vecs[-1])), dim=1)
            Gse = torch.softmax(torch.relu(torch.mm(self.se_emb_vecs[0], self.se_emb_vecs[-1])), dim=1)
        else:
            raise NotImplementedError

        # query event memory
        params_sp_enc, params_se_enc, params_sp_dec, params_se_dec = self.query_east_memo(ht_pair[0], ht_pair[1])

        # encoding
        x_t_sp = self.tcov_lin_sp(self.tcov_emb(t_x)).reshape(batch_size, seq_len, n_node, self.tcov_h_dim)
        x_t_se = self.tcov_lin_se(self.tcov_emb(t_x)).reshape(batch_size, seq_len, c_in, self.tcov_h_dim)
        _, ht_list_sp = self.encoder_sp(x_seq=torch.cat([x_seq, x_t_sp], dim=-1), G=Gsp, W3l=params_sp_enc, h0=None)
        _, ht_list_se = self.encoder_se(x_seq=torch.cat([x_seq.transpose(2, 3), x_t_se], dim=-1), G=Gse, W3l=params_se_enc, h0=None)
        ht_pair[0].data, ht_pair[1].data = ht_list_sp[-1].data, ht_list_se[-1].data      # update ht pair w/o breaking backpropagation

        # decoding
        go = torch.zeros((batch_size, n_node, c_in), device=x_seq.device)
        out = []
        for t in range(self.horizon):
            y_t_sp = self.tcov_lin_sp(self.tcov_emb(t_y[:, t, :])).reshape(batch_size, n_node, self.tcov_h_dim)
            y_t_se = self.tcov_lin_se(self.tcov_emb(t_y[:, t, :])).reshape(batch_size, c_in, self.tcov_h_dim)
            ht_sp, ht_list_sp = self.decoder_sp(xt=torch.cat([go, y_t_sp], dim=-1), G=Gsp, W3l=params_sp_dec, ht=ht_list_sp)  # (batch, n_node, h_dim)
            ht_se, ht_list_se = self.decoder_se(xt=torch.cat([go.transpose(1, 2), y_t_se], dim=-1), G=Gse, W3l=params_se_dec, ht=ht_list_se)  # (batch, c_in, h_dim)
            # output
            yt = torch.einsum('bnh,bch->bnc', ht_sp, self.proj(ht_se))
            out.append(yt)
            go = out[-1]

        out = torch.stack(out, dim=1)
        return torch.relu(out), ht_pair
