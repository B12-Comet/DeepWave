from abc import ABC, abstractmethod
from mimetypes import inited
from typing import Tuple, List, Dict
import torch.nn.init as init
import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.fft import fft, ifft, fftfreq 
import pywt
from pytorch_wavelets import DWT1DForward,DWT1DInverse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def forward_over_time(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

    def get_auc(
            self, queries: torch.Tensor, batch_size: int = 1000
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, begin, end)
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        all_scores, all_truth = [], []
        all_ts_ids = None
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                scores = self.forward_over_time(these_queries)
                all_scores.append(scores.cpu().numpy())
                if all_ts_ids is None:
                    all_ts_ids = torch.arange(0, scores.shape[1]).to(device)[None, :]
                assert not torch.any(torch.isinf(scores) + torch.isnan(scores)), "inf or nan scores"
                truth = (all_ts_ids <= these_queries[:, 4][:, None]) * (all_ts_ids >= these_queries[:, 3][:, None])
                all_truth.append(truth.cpu().numpy())
                b_begin += batch_size

        return np.concatenate(all_truth), np.concatenate(all_scores)

    def get_time_ranking(
            self, queries: torch.Tensor, filters: List[List[int]], chunk_size: int = -1
    ):
        """
        Returns filtered ranking for a batch of queries ordered by timestamp.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: ordered filters
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            q = self.get_queries(queries)
            targets = self.score(queries)
            while c_begin < self.sizes[2]:
                rhs = self.get_rhs(c_begin, chunk_size)
                scores = q @ rhs
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, (query, filter) in enumerate(zip(queries, filters)):
                    filter_out = filter + [query[2].item()]
                    if chunk_size < self.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        max_to_filter = max(filter_in_chunk + [-1])
                        assert max_to_filter < scores.shape[1], f"fuck {scores.shape[1]} {max_to_filter}"
                        scores[i, filter_in_chunk] = -1e6
                    else:
                        scores[i, filter_out] = -1e6
                ranks += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()

                c_begin += chunk_size
        return ranks



class DeepWave(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2,
            x: float = 1.0, y: float = 1.0,
            wavelet_type: str = 'db4', decomposition_level: int = 3  
    ):
        super(DeepWave, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.low_freq_weight = torch.nn.Parameter(torch.tensor(0.5))
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2 * rank, sparse=True),
            nn.Embedding(sizes[1], 2 * rank, sparse=True),
            nn.Embedding(sizes[2], 6 * rank, sparse=True)
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        # Text embedding
        self.textembedding = np.load('data/ICEWS14/ent_embed.npz')['embeddings']
        self.textembedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.textembedding), freeze=False)
        self.textembedding.weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.x = x
        self.y = y

        # Parameters related to the wavelet transform
        self.wavelet_type = wavelet_type
        self.decomposition_level = decomposition_level
        

    def wavelet_decompose(self, signal):
        """Decomposing signals using wavelet transforms"""
        if signal.shape[1] < 2 ** self.decomposition_level:
            pad_size = 2 ** self.decomposition_level - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, pad_size))

        signal_np = signal.cpu().detach().numpy()

        coeffs = []
        for i in range(signal.shape[0]):
            coeff = pywt.wavedec(signal_np[i], self.wavelet_type, level=self.decomposition_level)
            coeffs.append(coeff)

        return coeffs

    def wavelet_reconstruct(self, coeffs, original_length):
        """Reconstructing signals from the small-wave coefficient"""
        reconstructed = []
        for coeff in coeffs:
            
            if isinstance(coeff, list):
                approx_coeff = coeff[0]
                detail_coeffs = coeff[1:] if len(coeff) > 1 else []
                if len(detail_coeffs) > 0:
                    rec_signal = pywt.waverec([approx_coeff] + detail_coeffs, self.wavelet_type)
                else:
                    rec_signal = approx_coeff
            else:
                rec_signal = pywt.waverec(coeff, self.wavelet_type)

            if hasattr(rec_signal, '__len__'):
                rec_signal = rec_signal[:original_length]
            reconstructed.append(rec_signal)

        try:
            result = torch.tensor(np.array(reconstructed),
                                  device=coeffs[0][0].device if coeffs and coeffs[0] else 'cpu')
        except:
            processed_reconstructed = []
            for item in reconstructed:
                if not isinstance(item, np.ndarray):
                    item = np.array(item)
                processed_reconstructed.append(item)
            result = torch.tensor(np.array(processed_reconstructed), device='cpu')

        return result


    def decompose_relation_wavelet(self, rel_embedding):
        """Decomposing embedded relations into low-frequency and high-frequency components"""
        rel_real = rel_embedding[0]  
        rel_imag = rel_embedding[1]  
        original_length = rel_real.shape[1]
        coeffs_real = self.wavelet_decompose(rel_real)
        coeffs_imag = self.wavelet_decompose(rel_imag)

        # Separate low-frequency and high-frequency components - modify so that the first three tiers comprise low frequencies, with the final tier comprising high frequencies.
        low_freq_real_coeffs = [coeffs_real[i][:3].copy() for i in range(len(coeffs_real))]  
        high_freq_real_coeffs = [coeffs_real[i][3:].copy() for i in range(len(coeffs_real))]  

        low_freq_imag_coeffs = [coeffs_imag[i][:3].copy() for i in range(len(coeffs_imag))]
        high_freq_imag_coeffs = [coeffs_imag[i][3:].copy() for i in range(len(coeffs_imag))]

        # Reconstruct low-frequency components
        low_freq_signals_real = []
        for j in range(len(coeffs_real)):
            low_freq_coeff = low_freq_real_coeffs[j].copy()
            for k in range(len(coeffs_real[j]) - 3):
                low_freq_coeff.append(np.zeros_like(coeffs_real[j][k + 3]))
            low_freq_signals_real.append(low_freq_coeff)

        low_freq_component_real = self.wavelet_reconstruct(low_freq_signals_real, original_length)

        # Reconstruct high-frequency components
        high_freq_signals_real = []
        for j in range(len(coeffs_real)):
            high_freq_coeff = [np.zeros_like(coeffs_real[j][k]) for k in range(3)]  
            high_freq_coeff.extend(high_freq_real_coeffs[j])
            high_freq_signals_real.append(high_freq_coeff)

        high_freq_component_real = self.wavelet_reconstruct(high_freq_signals_real, original_length)

        # Apply the same treatment to the virtual section.
        low_freq_signals_imag = []
        for j in range(len(coeffs_imag)):
            low_freq_coeff = low_freq_imag_coeffs[j].copy()
            for k in range(len(coeffs_imag[j]) - 3):
                low_freq_coeff.append(np.zeros_like(coeffs_imag[j][k + 3]))
            low_freq_signals_imag.append(low_freq_coeff)

        low_freq_component_imag = self.wavelet_reconstruct(low_freq_signals_imag, original_length)

        high_freq_signals_imag = []
        for j in range(len(coeffs_imag)):
            high_freq_coeff = [np.zeros_like(coeffs_imag[j][k]) for k in range(3)]
            high_freq_coeff.extend(high_freq_imag_coeffs[j])
            high_freq_signals_imag.append(high_freq_coeff)

        high_freq_component_imag = self.wavelet_reconstruct(high_freq_signals_imag, original_length)

        return (low_freq_component_real, low_freq_component_imag), \
            (high_freq_component_real, high_freq_component_imag), \
            (low_freq_real_coeffs, low_freq_imag_coeffs), \
            (high_freq_real_coeffs, high_freq_imag_coeffs)
    


    

    def direct_interpolate_add(self, struct_emb, text_emb):
        """
        Directly use F.interpolate and then add them together.
        """
        target_dim = min(struct_emb.shape[1], text_emb.shape[1])
        struct_resized = F.interpolate(struct_emb.unsqueeze(1), size=target_dim, mode='linear',
                                       align_corners=True).squeeze(1)
        text_resized = F.interpolate(text_emb.unsqueeze(1), size=target_dim, mode='linear',
                                     align_corners=True).squeeze(1)
        return struct_resized + text_resized


    def score(self, x):
        ent_mod = self.direct_interpolate_add(self.embeddings[0].weight, self.textembedding.weight).to(device)
        lhs = ent_mod[x[:, 0]]
        rel = self.embeddings[1](x[:, 1])
        rhs = ent_mod[x[:, 2]]
        time = self.embeddings[2](x[:, 3]).to(device)
        x_real = self.x
        x_imag = self.x
        y_real = self.y
        y_imag = self.y

        lhs = (lhs[:, :self.rank] + x_real * time[:, 2 * self.rank:3 * self.rank],
               lhs[:, self.rank:] + x_imag * time[:, 3 * self.rank:4 * self.rank])
        rhs = (rhs[:, :self.rank] + y_real * time[:, 4 * self.rank:5 * self.rank],
               rhs[:, self.rank:] + y_imag * time[:, 5 * self.rank:6 * self.rank])

        
        rel_real, rel_imag = rel[:, :self.rank], rel[:, self.rank:2 * self.rank]
        time_real, time_imag = time[:, :self.rank], time[:, self.rank:2 * self.rank]

        # Frequency-domain decomposition relationship embedding
        (rel_low_real, rel_low_imag), (rel_high_real, rel_high_imag), _, _ = self.decompose_relation_wavelet(
            (rel_real, rel_imag))
        rel_low_real = rel_low_real.to(device)
        rel_low_imag = rel_low_imag.to(device)
        rel_high_imag = rel_high_imag.to(device)
        rel_high_real = rel_high_real.to(device)
        # Time smoothing and gradient computation
        time_smoothed_real = time_real.mean(dim=1, keepdim=True).to(device)
        time_smoothed_imag = time_imag.mean(dim=1, keepdim=True).to(device)
        time_gradient_real = torch.diff(time_real, dim=1, prepend=time_real[:, 0:1]).to(device)
        time_gradient_imag = torch.diff(time_imag, dim=1, prepend=time_imag[:, 0:1]).to(device)

        # Integrate low-frequency components (smoothing time) and high-frequency components (time gradient)
        rt_low_real = (rel_low_real + time_smoothed_real) * time_imag
        rt_low_imag = (rel_low_imag + time_smoothed_imag) * time_real
        rt_high_real = (rel_high_real + time_gradient_real) * time_imag
        rt_high_imag = (rel_high_imag + time_gradient_imag) * time_real

        low_weight = torch.sigmoid(self.low_freq_weight)  # Learnable parameters
        high_weight = 1 - low_weight
        

        rt_real = low_weight * rt_low_real + high_weight * rt_high_real
        rt_imag = low_weight * rt_low_imag + high_weight * rt_high_imag
        full_rel_real = rt_real
        full_rel_imag = rt_imag

        return torch.sum(
            (lhs[0] * full_rel_real - lhs[1] * full_rel_imag) * rhs[0] +
            (lhs[1] * full_rel_real + lhs[0] * full_rel_imag) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        ent_mod = self.direct_interpolate_add(self.embeddings[0].weight, self.textembedding.weight)

        lhs = ent_mod[x[:, 0]]
        rel = self.embeddings[1](x[:, 1])
        rhs = ent_mod[x[:, 2]]
        time = self.embeddings[2](x[:, 3])
       
        x_real=self.x
        x_imag=self.x
        y_real=self.y
        y_imag=self.y


        lhs = (lhs[:, :self.rank] + x_real * time[:, 2 * self.rank:3 * self.rank],
               lhs[:, self.rank:] + x_imag * time[:, 3 * self.rank:4 * self.rank])
        rhs = (rhs[:, :self.rank] + y_real * time[:, 4 * self.rank:5 * self.rank],
               rhs[:, self.rank:] + y_imag * time[:, 5 * self.rank:6 * self.rank])

        bias_t_r = y_real * time[:, 4 * self.rank:5 * self.rank]
        bias_t_i = y_imag * time[:, 5 * self.rank:6 * self.rank]

        # Acquire relational and temporal embeddings
        rel_real, rel_imag = rel[:, :self.rank], rel[:, self.rank:2 * self.rank]
        time_real, time_imag = time[:, :self.rank], time[:, self.rank:2 * self.rank]


        # Frequency domain decomposition
        (rel_low_real, rel_low_imag), (rel_high_real, rel_high_imag), low_freq, high_freq = self.decompose_relation_wavelet(
            (rel_real, rel_imag))
        rel_low_real=rel_low_real.to(device)
        rel_low_imag=rel_low_imag.to(device)
        rel_high_imag=rel_high_imag.to(device)
        rel_high_real=rel_high_real.to(device)

        # Time processing
        time_smoothed_real = time_real.mean(dim=1, keepdim=True)
        time_smoothed_imag = time_imag.mean(dim=1, keepdim=True)
        time_gradient_real = torch.diff(time_real, dim=1, prepend=time_real[:, 0:1])
        time_gradient_imag = torch.diff(time_imag, dim=1, prepend=time_imag[:, 0:1])

        # Frequency-domain–time fusion
        rt_low_real = (rel_low_real + time_smoothed_real) * time_imag
        rt_low_imag = (rel_low_imag + time_smoothed_imag) * time_real
        rt_high_real = (rel_high_real + time_gradient_real) * time_imag
        rt_high_imag = (rel_high_imag + time_gradient_imag) * time_real

        low_weight = torch.sigmoid(self.low_freq_weight)  # Learnable parameters
        high_weight = 1 - low_weight
        
        rt_real = low_weight * rt_low_real + high_weight * rt_high_real
        rt_imag = low_weight * rt_low_imag + high_weight * rt_high_imag
        full_rel_real = rt_real
        full_rel_imag = rt_imag

        right = ent_mod
        right = right[:, :self.rank], right[:, self.rank:]

        

        return (
                (lhs[0] * full_rel_real - lhs[1] * full_rel_imag) @ right[0].t() +
                torch.sum((lhs[0] * full_rel_real - lhs[1] * full_rel_imag) * bias_t_r, 1, keepdim=True) +
                (lhs[1] * full_rel_real + lhs[0] * full_rel_imag) @ right[1].t() +
                torch.sum((lhs[1] * full_rel_real + lhs[0] * full_rel_imag) * bias_t_i, 1, keepdim=True)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(full_rel_real ** 2 + full_rel_imag ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight


    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        ent_mod = self.direct_interpolate_add(self.embeddings[0].weight, self.textembedding.weight)
        return ent_mod.transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        ent_mod = self.direct_interpolate_add(self.embeddings[0].weight, self.textembedding.weight)
        lhs = ent_mod[queries[:, 0]]
        rel = self.embeddings[1](queries[:, 1]) 
        time = self.embeddings[2](queries[:, 3])
        
        x_real = self.x
        x_imag = self.x
        y_real = self.y
        y_imag = self.y

        lhs = (lhs[:, :self.rank] + x_real * time[:, 2 * self.rank:3 * self.rank],
               lhs[:, self.rank:] + x_imag * time[:, 3 * self.rank:4 * self.rank])


        bias_t_r = y_real * time[:, 4 * self.rank:5 * self.rank]
        bias_t_i = y_imag * time[:, 5 * self.rank:6 * self.rank]

        # Acquire relational and temporal embeddings
        rel_real, rel_imag = rel[:, :self.rank], rel[:, self.rank:2 * self.rank]
        time_real, time_imag = time[:, :self.rank], time[:, self.rank:2 * self.rank]

        # Frequency domain decomposition
        (rel_low_real, rel_low_imag), (rel_high_real, rel_high_imag), low_freq, high_freq = self.decompose_relation_wavelet(
            (rel_real, rel_imag))
        rel_low_real = rel_low_real.to(device)
        rel_low_imag = rel_low_imag.to(device)
        rel_high_imag = rel_high_imag.to(device)
        rel_high_real = rel_high_real.to(device)
        # Time processing
        time_smoothed_real = time_real.mean(dim=1, keepdim=True)
        time_smoothed_imag = time_imag.mean(dim=1, keepdim=True)
        time_gradient_real = torch.diff(time_real, dim=1, prepend=time_real[:, 0:1])
        time_gradient_imag = torch.diff(time_imag, dim=1, prepend=time_imag[:, 0:1])

        # Frequency-domain–time fusion
        rt_low_real = (rel_low_real + time_smoothed_real) * time_imag
        rt_low_imag = (rel_low_imag + time_smoothed_imag) * time_real
        rt_high_real = (rel_high_real + time_gradient_real) * time_imag
        rt_high_imag = (rel_high_imag + time_gradient_imag) * time_real
        low_weight = torch.sigmoid(self.low_freq_weight)  # Learnable parameters
        high_weight = 1 - low_weight
    
        rt_real = low_weight * rt_low_real + high_weight * rt_high_real
        rt_imag = low_weight * rt_low_imag + high_weight * rt_high_imag
        full_rel_real = rt_real
        full_rel_imag = rt_imag

        right = ent_mod
        right = right[:, :self.rank], right[:, self.rank:]

        
        return torch.cat([
            (lhs[0] * full_rel_real - lhs[1] * full_rel_imag),
            (lhs[1] * full_rel_real + lhs[0] * full_rel_imag)
        ], 1)


