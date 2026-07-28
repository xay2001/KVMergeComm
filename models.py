from typing import Literal, Optional
import copy
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import DynamicCache, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from model_attn import LlamaAttentionTracer, Qwen2AttentionTracer, Gemma3AttentionTracer
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel, repeat_kv
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention

def get_layer_map(L_A, L_B):
    layer_map = {}
    for l_a in range(L_A):
        layer_map[l_a] = round( (l_a + 0.5) * L_B / L_A - 0.5 )
    return layer_map

class CVCommunicator(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_A: PreTrainedModel,
        model_B: PreTrainedModel,
        layer_from: int,
        layer_to: int,
        top_layers: float = 0.0,
        layers_list: list[int] = [],
        apply_attn_tracer: bool = False,
        shift_back: bool = False,
        merge: bool = False,
        merge_ratio: float = 0.2,
        merge_sink: int = 4,
        merge_recent: int = 8,
        merge_mode: str = "merge",
        score_mode: str = "value_norm",
        recv_window: int = 0,
        query_sketch_mode: str = "bf16",
        receiver_layer_agg: str = "identity",
        budget_mode: str = "uniform",
        budget_min: float = 0.05,
        budget_max: float = 0.5,
        budget_tau: float = 1.0,
        budget_floor: float = 0.02,
        coverage_tau: float = 0.90,
        coverage_scale: float = 1.0,
        coverage_tau_mode: str = "fixed",
        coverage_tau_min: float = 0.80,
        coverage_tau_max: float = 0.95,
        query_condition_mode: str = "correct",
        budget_replay_tolerance: float = 1e-3,
    ) -> None:
        super().__init__(model_B.config)
        self.A = model_A
        self.B = model_B
        self.layer_from = layer_from
        self.layer_to = layer_to
        self.apply_attn_tracer = apply_attn_tracer
        self.shift_back = shift_back
        self.merge = merge
        self.merge_ratio = merge_ratio
        self.merge_sink = merge_sink
        self.merge_recent = merge_recent
        self.merge_mode = merge_mode  # "merge" (normalized value merge) or "evict" (drop only)
        self.score_mode = score_mode  # "value_norm", "random", "receiver", or receiver-aware ablation variants
        self.recv_window = recv_window  # 0 = all question tokens; >0 = only last N (SnapKV-style observation window)
        self.query_sketch_mode = str(query_sketch_mode).lower()
        if self.query_sketch_mode not in {"bf16", "int8", "token_ids"}:
            raise ValueError(f"unknown query_sketch_mode={query_sketch_mode}")
        self.receiver_layer_agg = receiver_layer_agg  # identity | last | mean | topK | lastK
        self.token_importance = None  # filled per-sample by compute_receiver_importance for receiver-aware score modes
        # budget-aware allocation (Step 1): how the per-query / per-layer keep ratio is set.
        #   uniform      -> every layer keeps self.merge_ratio (original ReKV behaviour)
        #   query        -> per-query total budget B(Q) from importance entropy, uniform across layers
        #   layer        -> fixed total budget self.merge_ratio, softmax-allocated across layers by importance
        #   query+layer  -> both: B(Q) total, softmax-allocated across layers
        #   coverage     -> per-layer budget from receiver-attention coverage threshold
        self.budget_mode = budget_mode
        self.budget_min = budget_min
        self.budget_max = budget_max
        self.budget_tau = budget_tau
        self.budget_floor = budget_floor
        self.coverage_tau = coverage_tau
        self.coverage_scale = coverage_scale
        self.coverage_tau_mode = coverage_tau_mode
        self.coverage_tau_min = coverage_tau_min
        self.coverage_tau_max = coverage_tau_max
        self.query_condition_mode = str(query_condition_mode).lower()
        self.budget_replay_tolerance = float(budget_replay_tolerance)
        self.replay_target_budget = None
        self.replay_layer_budget = {}
        self.layer_budget = {}        # {layer_idx: r_l}, recomputed per query
        self.layer_coverage_target = {}
        self.layer_coverage_achieved = {}
        self.last_query_budget = None # B(Q) for the most recent query
        self.last_coverage_target = None
        self.last_coverage_achieved = None
        self.last_coverage_satisfied_ratio = None
        self.last_kept_ratio = None   # actual transmitted KV fraction for the most recent query
        self.last_kv_cost = None      # payload stats from the most recent cache preparation
        self.last_query_sketch_cost = {}  # B->A query-sketch payload for the most recent query
        self.last_protocol_timing = {}
        self.protocol_version = self._protocol_version()
        self._merge_logged = False
        for p in self.A.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False

        if hasattr(self.A.config, "num_hidden_layers"):
            self.A_num_layers = self.A.config.num_hidden_layers
        elif hasattr(self.A.config, "text_config") and hasattr(self.A.config.text_config, "num_hidden_layers"):
            self.A_num_layers = self.A.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.A.config}")
        if hasattr(self.B.config, "num_hidden_layers"):
            self.B_num_layers = self.B.config.num_hidden_layers
        elif hasattr(self.B.config, "text_config") and hasattr(self.B.config.text_config, "num_hidden_layers"):
            self.B_num_layers = self.B.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.B.config}")

        if layers_list[0] != -1:
            self.layers_list = layers_list
        elif top_layers > 0:
            self.layers_list = list(range(0, self.A_num_layers)) # set all layers at first
        else:
            self.layers_list = list(range(self.layer_from, self.layer_to + 1))

        self.layer_map = get_layer_map(self.A_num_layers, self.B_num_layers)

        if apply_attn_tracer:
            self.B_attn_weights = {}
            self.apply_B_attn_tracer()
            if (
                self.query_sketch_mode == "token_ids"
                and self.score_mode != "receiver_oracle"
            ) or self.query_condition_mode == "sender_context_q":
                self.apply_A_attn_tracer()

        logging.info(f"CVCommunicator initialized")

    def _protocol_version(self) -> str:
        if self.query_condition_mode == "sender_context_q":
            return "sender_context_q_local_v1"
        if self.score_mode == "receiver_oracle":
            return "full_kv_oracle_v1"
        if self.score_mode in {
            "receiver",
            "receiver_x_value_norm",
            "receiver_value_norm",
            "receiver_recency",
            "receiver_recency_prior",
        }:
            return f"query_sketch_{self.query_sketch_mode}_v1"
        return "query_agnostic_kv_v1"

    def set_replay_budget(self, target: float, past_key_values) -> None:
        """Build per-layer ratios whose realized total-KV ratio matches target."""
        target = min(max(float(target), 0.0), 1.0)
        lengths = [int(key.shape[-2]) for key in past_key_values.key_cache]
        if not lengths:
            raise ValueError("budget replay requires a non-empty KV cache")
        total_tokens = sum(lengths)
        desired_total = int(round(target * total_tokens))
        minimum = [
            min(self.merge_sink + self.merge_recent, length)
            for length in lengths[1:]
        ]
        kept = minimum[:]
        desired_compressed = min(
            max(desired_total - lengths[0], sum(minimum)),
            sum(lengths[1:]),
        )
        remaining = desired_compressed - sum(kept)
        while remaining > 0:
            active = [
                index
                for index, length in enumerate(lengths[1:])
                if kept[index] < length
            ]
            if not active:
                break
            share = max(remaining // len(active), 1)
            for index in active:
                capacity = lengths[index + 1] - kept[index]
                addition = min(share, capacity, remaining)
                kept[index] += addition
                remaining -= addition
                if remaining == 0:
                    break
        self.replay_layer_budget = {
            layer_idx: kept[layer_idx - 1] / lengths[layer_idx]
            for layer_idx in range(1, len(lengths))
        }
        self.replay_target_budget = target

    @staticmethod
    def _sync_tensor_device(tensor: torch.Tensor):
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def _apply_attn_tracer(self, model):
        if hasattr(model.model, "language_model"):
            layers = model.model.language_model.layers
        else:
            layers = model.model.layers
        for i, block in enumerate(layers):
            old = block.self_attn
            device = next(old.parameters()).device
            dtype  = next(old.parameters()).dtype
            if type(old) in (Qwen2AttentionTracer, LlamaAttentionTracer, Gemma3AttentionTracer):
                continue
            if type(old) is Qwen2Attention:
                new = Qwen2AttentionTracer(old.config, old.layer_idx).to(device, dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                block.self_attn = new
            elif type(old) is LlamaAttention:
                new = LlamaAttentionTracer(old.config, old.layer_idx).to(device, dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                block.self_attn = new
            elif type(old) is Gemma3Attention:
                new = Gemma3AttentionTracer(old.config, old.layer_idx).to(device, dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                block.self_attn = new
            else:
                raise ValueError(f"Unsupported attention module: {type(old)}")

    def apply_B_attn_tracer(self):
        self._apply_attn_tracer(self.B)

    def apply_A_attn_tracer(self):
        self._apply_attn_tracer(self.A)

    def prepare_key_cache(self, past_key_values):
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
        assert len(key_cache) == len(self.layer_map), "key_cache and layer_map must have the same length"
        self._sync_tensor_device(key_cache[0])
        cache_prepare_start = time.perf_counter()
        past_key_values_new = DynamicCache()
        self.layer_coverage_achieved = {}
        self.last_coverage_achieved = None
        self.last_coverage_satisfied_ratio = None
        kept_tokens, total_tokens = 0, 0
        kept_kv_elements, total_kv_elements = 0, 0
        kept_kv_bytes, total_kv_bytes = 0, 0
        for i in range(len(key_cache)): # i is the layer index of model A
            total_tokens += key_cache[i].shape[-2]
            total_kv_elements += key_cache[i].numel() + value_cache[i].numel()
            total_kv_bytes += (
                key_cache[i].numel() * key_cache[i].element_size()
                + value_cache[i].numel() * value_cache[i].element_size()
            )
            if self.merge:
                # Merge-then-Communicate: keep all layers, compress tokens within each
                # layer by merging (instead of dropping whole layers). Layer 0 is kept
                # full to anchor the receiver's position indexing.
                if i == 0:
                    key_cache_i, value_cache_i = key_cache[i], value_cache[i]
                else:
                    key_cache_i, value_cache_i = self.compress_merge_layer(key_cache[i], value_cache[i], layer_idx=i)
                past_key_values_new.update(key_cache_i, value_cache_i, self.layer_map[i])
            elif i in self.layers_list or i == 0:
                key_cache_i, value_cache_i = key_cache[i], value_cache[i]
                past_key_values_new.update(key_cache_i, value_cache_i, self.layer_map[i])
            else:
                # keep the first token due to attention sink
                key_cache_i = key_cache[i][:, :, :1, :]
                value_cache_i = value_cache[i][:, :, :1, :]
                past_key_values_new.update(key_cache_i, value_cache_i, self.layer_map[i])
            kept_tokens += key_cache_i.shape[-2]
            kept_kv_elements += key_cache_i.numel() + value_cache_i.numel()
            kept_kv_bytes += (
                key_cache_i.numel() * key_cache_i.element_size()
                + value_cache_i.numel() * value_cache_i.element_size()
            )
        self.last_kept_ratio = kept_tokens / total_tokens if total_tokens else None
        if self.replay_target_budget is not None and self.last_kept_ratio is not None:
            error = abs(self.last_kept_ratio - self.replay_target_budget)
            if error > self.budget_replay_tolerance:
                raise RuntimeError(
                    "budget replay mismatch: "
                    f"target={self.replay_target_budget:.6f}, "
                    f"actual={self.last_kept_ratio:.6f}, "
                    f"error={error:.6f}, "
                    f"tolerance={self.budget_replay_tolerance:.6f}"
                )
        self.last_kv_cost = {
            "protocol_version": self.protocol_version,
            "kv_tokens_sent": int(kept_tokens),
            "kv_tokens_total": int(total_tokens),
            "kv_token_ratio": float(kept_tokens / total_tokens) if total_tokens else None,
            "kv_elements_sent": int(kept_kv_elements),
            "kv_elements_total": int(total_kv_elements),
            "kv_element_ratio": float(kept_kv_elements / total_kv_elements) if total_kv_elements else None,
            "kv_bytes_sent": int(kept_kv_bytes),
            "kv_bytes_total": int(total_kv_bytes),
            "kv_byte_ratio": float(kept_kv_bytes / total_kv_bytes) if total_kv_bytes else None,
            **self.last_query_sketch_cost,
        }
        query_sketch_bytes = int(self.last_query_sketch_cost.get("query_sketch_bytes", 0))
        query_sketch_metadata_bytes = int(
            self.last_query_sketch_cost.get("query_sketch_metadata_bytes", 0)
        )
        oracle_full_kv_bytes = int(self.last_query_sketch_cost.get("oracle_full_kv_bytes", 0))
        # Conservative wire-format accounting. Each transmitted KV token carries
        # one uint32 source-position index. Each layer descriptor carries five
        # uint32 values: source layer, receiver layer, token count, KV heads, and
        # head dimension; the message header is two uint32 values.
        index_bytes = int(kept_tokens * 4)
        kv_metadata_bytes = int(8 + len(key_cache) * 20)
        a_to_b_bytes = int(kept_kv_bytes + index_bytes + kv_metadata_bytes)
        b_to_a_bytes = int(query_sketch_bytes + query_sketch_metadata_bytes)
        self.last_kv_cost.update(
            {
                "selection_index_bytes": index_bytes,
                "kv_metadata_bytes": kv_metadata_bytes,
                "a_to_b_communication_bytes": a_to_b_bytes,
                "b_to_a_communication_bytes": b_to_a_bytes,
                "communication_metadata_bytes": int(
                    index_bytes + kv_metadata_bytes + query_sketch_metadata_bytes
                ),
                "total_communication_bytes": (
                    int(oracle_full_kv_bytes + kv_metadata_bytes)
                    if oracle_full_kv_bytes > 0
                    else int(a_to_b_bytes + b_to_a_bytes)
                ),
            }
        )
        if self.layer_coverage_achieved:
            achieved = list(self.layer_coverage_achieved.values())
            satisfied = [
                float(self.layer_coverage_achieved[l] + 1e-6 >= self.layer_coverage_target.get(l, 0.0))
                for l in self.layer_coverage_achieved
            ]
            self.last_coverage_achieved = sum(achieved) / len(achieved)
            self.last_coverage_satisfied_ratio = sum(satisfied) / len(satisfied)
            self.last_kv_cost.update(
                {
                    "coverage_target": self.last_coverage_target,
                    "coverage_achieved": self.last_coverage_achieved,
                    "coverage_satisfied_layer_ratio": self.last_coverage_satisfied_ratio,
                }
            )
        self._sync_tensor_device(key_cache[0])
        t_sender_compress = time.perf_counter() - cache_prepare_start
        self.last_protocol_timing["t_sender_compress"] = float(t_sender_compress)
        self.last_kv_cost["t_sender_compress"] = float(t_sender_compress)
        return past_key_values_new

    @torch.no_grad()
    def compress_merge_layer(self, K: torch.Tensor, V: torch.Tensor, layer_idx: int = -1):
        """Compress one layer's KV from L tokens down to k via importance selection +
        value merging (CaM/DualKV style). Retained tokens keep their original keys
        (RoPE intact); evicted tokens' values are distributed into the retained ones
        by key-similarity, so information is merged rather than dropped.

        K, V: [B, H_kv, L, D]
        """
        B, H, L, D = K.shape
        r_eff = self._effective_ratio(layer_idx)
        k = max(int(round(r_eff * L)), self.merge_sink + self.merge_recent)
        if k >= L:
            if self.budget_mode in {"coverage", "strict_coverage"}:
                self.layer_coverage_achieved[layer_idx] = 1.0
            return K, V

        # token importance score
        value_norm = V.float().norm(dim=-1).mean(dim=1)[0]  # [L]
        if (
            self.score_mode in {"receiver", "receiver_oracle", "receiver_x_value_norm", "receiver_value_norm", "receiver_recency", "receiver_recency_prior"}
            and self.token_importance is not None
            and layer_idx in self.token_importance
            and self.token_importance[layer_idx].numel() == L
        ):
            # receiver-aware: how much B's question attends to each A-context token
            recv_imp = self.token_importance[layer_idx].to(K.device).float().clamp_min(0)  # [L]
            if self.score_mode in {"receiver_x_value_norm", "receiver_value_norm"}:
                vn = value_norm.clamp_min(0)
                imp = (recv_imp / (recv_imp.mean() + 1e-12)) * (vn / (vn.mean() + 1e-12))
            elif self.score_mode in {"receiver_recency", "receiver_recency_prior"}:
                recency = torch.linspace(0.0, 1.0, L, device=K.device, dtype=torch.float32)
                imp = recv_imp + 0.1 * recv_imp.mean().clamp_min(1e-12) * recency
            else:
                imp = recv_imp.clone()
        elif self.score_mode == "random":
            # Random-token baseline for selection ablations.
            imp = torch.rand(L, device=K.device, dtype=torch.float32)
        else:
            # query-agnostic proxy: value vector L2-norm
            imp = value_norm
        big = torch.finfo(imp.dtype).max
        if self.merge_sink > 0:
            imp[: self.merge_sink] = big
        if self.merge_recent > 0:
            imp[L - self.merge_recent :] = big

        keep = torch.topk(imp, k).indices
        keep, _ = torch.sort(keep)
        if (
            self.budget_mode in {"coverage", "strict_coverage"}
            and self.token_importance is not None
            and layer_idx in self.token_importance
        ):
            coverage_scores = self.token_importance[layer_idx].to(K.device).float().clamp_min(0)
            coverage_total = coverage_scores.sum()
            if coverage_total > 0:
                self.layer_coverage_achieved[layer_idx] = float(
                    coverage_scores[keep].sum() / coverage_total
                )
        evict_mask = torch.ones(L, dtype=torch.bool, device=K.device)
        evict_mask[keep] = False
        evict = torch.nonzero(evict_mask, as_tuple=False).squeeze(-1)

        Kk = K[:, :, keep, :].contiguous()
        Vk = V[:, :, keep, :].clone()
        if self.merge_mode == "merge" and evict.numel() > 0:
            Ke = K[:, :, evict, :]
            Ve = V[:, :, evict, :]
            sim = torch.matmul(Ke, Kk.transpose(-1, -2)) / (D ** 0.5)  # [B,H,e,k]
            w = torch.softmax(sim.float(), dim=-1)  # [B,H,e,k], each evicted token distributes mass 1 over kept
            agg = torch.matmul(w.transpose(-1, -2), Ve.float())  # [B,H,k,D] sum of merged-in values
            denom = 1.0 + w.sum(dim=-2, keepdim=True).transpose(-1, -2)  # [B,H,k,1] self + received weights
            Vk = ((Vk.float() + agg) / denom).to(V.dtype)  # normalized convex combination -> no magnitude blow-up

        if not self._merge_logged:
            logging.info(f"[merge] mode={self.merge_mode} score={self.score_mode} budget={self.budget_mode} layer compress: L={L} -> k={k} (r_eff={r_eff:.3f}, base_ratio={self.merge_ratio}, sink={self.merge_sink}, recent={self.merge_recent})")
            self._merge_logged = True
        return Kk, Vk.contiguous()

    def _effective_ratio(self, layer_idx: int) -> float:
        """Per-layer keep ratio. Falls back to the global merge_ratio unless a
        budget-aware allocation has been computed for this query/layer."""
        if layer_idx in self.replay_layer_budget:
            return self.replay_layer_budget[layer_idx]
        if self.budget_mode == "uniform" or not self.layer_budget:
            return self.merge_ratio
        return self.layer_budget.get(layer_idx, self.merge_ratio)

    def _query_total_budget(self) -> float:
        """Per-query total budget B(Q) from the receiver-importance distribution.
        Diffuse importance (high entropy) -> evidence spread over many tokens ->
        needs a larger budget; concentrated importance -> small budget suffices.
            B = budget_min + (budget_max - budget_min) * mean_l(H_l / log L_l)
        """
        Hns = []
        for l, s in self.token_importance.items():
            if l == 0:
                continue
            s = s.float()
            tot = s.sum()
            if tot <= 0 or s.numel() < 2:
                continue
            p = s / (tot + 1e-9)
            H = -(p * (p + 1e-12).log()).sum()
            Hns.append((H / math.log(s.numel())).clamp(0.0, 1.0).item())
        if not Hns:
            return self.merge_ratio
        Hmean = sum(Hns) / len(Hns)
        return self.budget_min + (self.budget_max - self.budget_min) * Hmean

    def _coverage_ratio(self, s: torch.Tensor, target: float | None = None) -> float:
        """Budget from receiver-evidence coverage.

        Let p_i be normalized receiver attention mass over A-context tokens. We
        keep the smallest top-k set whose cumulative mass reaches coverage_tau,
        then optionally scale and clamp the resulting k/L ratio. Unlike the
        entropy/query predictor, this does not learn or predict task difficulty;
        the budget is derived from an interpretable fidelity target.
        """
        s = s.float().clamp_min(0)
        L = s.numel()
        if L < 2 or float(s.sum()) <= 0:
            return self.merge_ratio
        p = s / s.sum().clamp_min(1e-9)
        sp = torch.sort(p, descending=True).values
        csum = torch.cumsum(sp, dim=0)
        if target is None:
            target = self.coverage_tau
        thr = min(max(float(target), 0.0), 1.0)
        idx = int(torch.searchsorted(csum, torch.tensor(thr, device=csum.device)))
        idx = min(idx, L - 1)
        raw = ((idx + 1) / L) * float(self.coverage_scale)
        lo = max(float(self.budget_min), float(self.budget_floor))
        hi = float(self.budget_max)
        if hi < lo:
            hi = lo
        return float(min(max(raw, lo), hi))

    def _adaptive_coverage_tau(self, layers: list[int]) -> float:
        """Per-query target from mean normalized attention entropy.

        Concentrated evidence uses ``coverage_tau_min``; diffuse evidence moves
        smoothly toward ``coverage_tau_max``.
        """
        entropies = []
        for layer_idx in layers:
            s = self.token_importance[layer_idx].float().clamp_min(0)
            if s.numel() < 2 or float(s.sum()) <= 0:
                continue
            p = s / s.sum().clamp_min(1e-9)
            entropy = -(p * (p + 1e-12).log()).sum() / math.log(p.numel())
            entropies.append(float(entropy.clamp(0.0, 1.0)))
        diffuseness = sum(entropies) / len(entropies) if entropies else 0.5
        lo = min(max(float(self.coverage_tau_min), 0.0), 1.0)
        hi = min(max(float(self.coverage_tau_max), lo), 1.0)
        return lo + (hi - lo) * diffuseness

    def _strict_coverage_ratio(self, s: torch.Tensor, target: float) -> float:
        """Minimum keep ratio that truly covers ``target`` attention mass.

        Mandatory sink/recent tokens are included before selecting the smallest
        additional top-mass set. No scale or budget clamp is applied.
        """
        s = s.float().clamp_min(0)
        length = s.numel()
        if length == 0 or float(s.sum()) <= 0:
            return 1.0
        p = s / s.sum().clamp_min(1e-9)
        forced = torch.zeros(length, dtype=torch.bool, device=s.device)
        if self.merge_sink > 0:
            forced[: min(self.merge_sink, length)] = True
        if self.merge_recent > 0:
            forced[max(length - self.merge_recent, 0) :] = True

        forced_count = int(forced.sum())
        forced_mass = float(p[forced].sum())
        needed = max(min(float(target), 1.0) - forced_mass, 0.0)
        extra_count = 0
        if needed > 0 and forced_count < length:
            remaining = torch.sort(p[~forced], descending=True).values
            csum = torch.cumsum(remaining, dim=0)
            idx = int(torch.searchsorted(csum, torch.tensor(needed, device=csum.device)))
            extra_count = min(idx + 1, remaining.numel())
        return float((forced_count + extra_count) / length)

    def _compute_budget(self):
        """Recompute the per-layer keep ratios for the current query. Called right
        after receiver importance is available. Requires score_mode=='receiver'."""
        self.layer_budget = {}
        self.last_query_budget = None
        if self.budget_mode == "uniform" or not self.token_importance:
            return
        comp = [l for l in sorted(self.token_importance.keys()) if l != 0]
        if not comp:
            return
        self.layer_coverage_target = {}
        self.last_coverage_target = None
        if self.budget_mode == "strict_coverage":
            if str(self.coverage_tau_mode).lower() == "adaptive":
                target = self._adaptive_coverage_tau(comp)
            else:
                target = min(max(float(self.coverage_tau), 0.0), 1.0)
            self.last_coverage_target = float(target)
            self.layer_coverage_target = {l: float(target) for l in comp}
            self.layer_budget = {
                l: self._strict_coverage_ratio(self.token_importance[l], target)
                for l in comp
            }
            self.last_query_budget = sum(self.layer_budget.values()) / len(self.layer_budget)
            return
        if self.budget_mode == "coverage":
            if str(self.coverage_tau_mode).lower() == "adaptive":
                target = self._adaptive_coverage_tau(comp)
            else:
                target = min(max(float(self.coverage_tau), 0.0), 1.0)
            self.last_coverage_target = float(target)
            self.layer_coverage_target = {l: self.last_coverage_target for l in comp}
            self.layer_budget = {
                l: self._coverage_ratio(self.token_importance[l], target)
                for l in comp
            }
            self.last_query_budget = sum(self.layer_budget.values()) / len(self.layer_budget)
            return
        B = self._query_total_budget() if self.budget_mode in ("query", "query+layer") else self.merge_ratio
        self.last_query_budget = float(B)
        if self.budget_mode in ("layer", "query+layer"):
            # layer importance: top-10% mean of receiver attention mass (concentration-robust)
            I = []
            for l in comp:
                s = self.token_importance[l].float()
                kk = max(1, int(round(0.1 * s.numel())))
                I.append(torch.topk(s, min(kk, s.numel())).values.mean())
            Ivec = torch.stack(I)
            w = torch.softmax(Ivec / self.budget_tau, dim=0)  # sums to 1 over layers
            r = (w * len(comp) * B).clamp(self.budget_floor, 1.0)  # mean over layers ~= B
            self.layer_budget = {l: float(r[i]) for i, l in enumerate(comp)}
        else:  # query-only: same B on every compressible layer
            self.layer_budget = {l: float(B) for l in comp}

    def _receiver_layers(self):
        if hasattr(self.B.model, "language_model"):
            return self.B.model.language_model.layers
        return self.B.model.layers

    def _sender_layers(self):
        if hasattr(self.A.model, "language_model"):
            return self.A.model.language_model.layers
        return self.A.model.layers

    @torch.no_grad()
    def compute_receiver_importance(self, input_ids_B, out_A_past_key_values):
        """Deployable receiver-query-aware scoring from a B->A Q sketch.

        ``bf16`` and ``int8`` encode B's layer-wise query vectors; ``token_ids``
        sends the observation-window token IDs once and lets A encode them
        locally. A always scores against its local context keys, so B never
        receives the full A cache during token selection.
        """
        assert self.apply_attn_tracer, "receiver scoring needs apply_attn_tracer=True"
        assert len(out_A_past_key_values.key_cache) == self.A_num_layers
        self.last_protocol_timing = {}

        ctx_len = [out_A_past_key_values.key_cache[i].shape[-2] for i in range(self.A_num_layers)]
        if len(set(ctx_len)) != 1:
            raise ValueError(f"query-sketch scoring requires equal uncompressed A context lengths, got {sorted(set(ctx_len))}")
        context_length = int(ctx_len[0])
        query_length = int(input_ids_B.shape[-1])
        query_input_ids = input_ids_B
        query_start = 0
        if self.query_sketch_mode == "token_ids" and self.recv_window > 0:
            query_start = max(query_length - self.recv_window, 0)
            query_input_ids = input_ids_B[:, query_start:]
        position_ids = torch.arange(
            context_length + query_start,
            context_length + query_length,
            device=input_ids_B.device,
            dtype=torch.long,
        ).unsqueeze(0)

        # Query-only prefill: Q-vector modes run B, while token-ID mode sends
        # the IDs first and has A encode them locally.
        query_encoder = self.A if self.query_sketch_mode == "token_ids" else self.B
        self._sync_tensor_device(input_ids_B)
        query_prefill_start = time.perf_counter()
        _ = query_encoder(
            input_ids=query_input_ids,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        self._sync_tensor_device(input_ids_B)
        query_prefill_time = float(time.perf_counter() - query_prefill_start)
        if self.query_sketch_mode == "token_ids":
            self.last_protocol_timing["t_a_query_encode"] = query_prefill_time
            self.last_protocol_timing["t_b_query_prefill"] = 0.0
        else:
            self.last_protocol_timing["t_b_query_prefill"] = query_prefill_time
            self.last_protocol_timing["t_a_query_encode"] = 0.0

        layers = self._sender_layers() if self.query_sketch_mode == "token_ids" else self._receiver_layers()
        per_layer_importance = {}
        sketch_elements = 0
        sketch_bytes = 0
        sketch_scale_bytes = 0
        sketch_tokens = None

        self._sync_tensor_device(input_ids_B)
        sender_score_start = time.perf_counter()
        for al in range(self.A_num_layers):
            query_layer = al if self.query_sketch_mode == "token_ids" else self.layer_map[al]
            block = layers[query_layer]
            attn_inputs = block.self_attn.attn_inputs
            if attn_inputs is None:
                raise RuntimeError(f"query tracer did not capture layer {query_layer}")
            query = attn_inputs["query"]
            if self.query_sketch_mode != "token_ids" and self.recv_window > 0 and query.shape[-2] > self.recv_window:
                query = query[:, :, -self.recv_window :, :]
            key = out_A_past_key_values.key_cache[al]
            if self.query_sketch_mode == "bf16":
                wire_query = query.to(torch.bfloat16).contiguous()
                sketch_elements += wire_query.numel()
                sketch_bytes += wire_query.numel() * wire_query.element_size()
                query = wire_query.to(key.dtype)
            elif self.query_sketch_mode == "int8":
                max_abs = query.float().abs().amax().clamp_min(1e-12)
                scale = max_abs / 127.0
                query_int8 = torch.clamp(torch.round(query.float() / scale), -127, 127).to(torch.int8)
                query = (query_int8.float() * scale).to(key.dtype).contiguous()
                sketch_elements += query_int8.numel()
                sketch_bytes += query_int8.numel()
                sketch_scale_bytes += 4
            else:
                query = query.to(key.dtype).contiguous()

            if query.shape[-1] != key.shape[-1]:
                raise ValueError(
                    f"query-sketch head_dim mismatch at A layer {al}/query layer {query_layer}: "
                    f"Q={query.shape[-1]}, K={key.shape[-1]}"
                )
            expected_q_heads = key.shape[1] * block.self_attn.num_key_value_groups
            if query.shape[1] != expected_q_heads:
                raise ValueError(
                    f"query-sketch head mismatch at A layer {al}/query layer {query_layer}: "
                    f"Q heads={query.shape[1]}, expected={expected_q_heads} "
                    f"from A KV heads={key.shape[1]}"
                )

            # This computation is sender-local: Q sketch x local K_A.
            attn_weights = eager_attention_forward_without_value(
                block.self_attn,
                query=query,
                key=key,
                attention_mask=None,
                scaling=block.self_attn.scaling,
            )
            per_layer_importance[al] = attn_weights.float().mean(dim=1)[0].sum(dim=0).contiguous()
            sketch_tokens = int(query.shape[-2])

        self._sync_tensor_device(input_ids_B)
        self.last_protocol_timing["t_sender_score"] = float(
            time.perf_counter() - sender_score_start
        )
        if self.query_sketch_mode == "token_ids":
            # Token IDs are transmitted once as uint32, not once per layer.
            sketch_elements = int(query_input_ids.numel())
            sketch_bytes = int(query_input_ids.numel() * 4)
            sketch_layers = 1
            sketch_metadata_bytes = 8
        else:
            # Two uint32 values in the message header and four uint32 values per
            # layer: layer id, query tokens, query heads, and head dimension.
            sketch_layers = self.A_num_layers
            sketch_metadata_bytes = int(8 + self.A_num_layers * 16 + sketch_scale_bytes)
        self.last_query_sketch_cost = {
            "protocol_version": self.protocol_version,
            "query_sketch_mode": self.query_sketch_mode,
            "query_sketch_tokens": int(sketch_tokens or 0),
            "query_sketch_layers": int(sketch_layers),
            "query_sketch_elements": int(sketch_elements),
            "query_sketch_bytes": int(sketch_bytes),
            "query_sketch_scale_bytes": int(sketch_scale_bytes),
            "query_sketch_metadata_bytes": sketch_metadata_bytes,
        }
        self._sync_tensor_device(input_ids_B)
        budget_start = time.perf_counter()
        self.token_importance = self._aggregate_receiver_importance(per_layer_importance)
        self._compute_budget()
        self._sync_tensor_device(input_ids_B)
        self.last_protocol_timing["t_budget_compute"] = float(
            time.perf_counter() - budget_start
        )

    @torch.no_grad()
    def compute_sender_context_importance(self, out_A_past_key_values):
        """Score A's KV with A's own final context-query vectors.

        This is a query-free sender-derived causal control: no receiver query or
        B-side query sketch participates in token selection.
        """
        assert self.apply_attn_tracer, "sender-context scoring needs an attention tracer"
        assert len(out_A_past_key_values.key_cache) == self.A_num_layers
        self.last_protocol_timing = {
            "t_b_query_prefill": 0.0,
            "t_a_query_encode": 0.0,
        }
        layers = self._sender_layers()
        per_layer_importance = {}
        score_start = time.perf_counter()
        for layer_idx, block in enumerate(layers):
            attn_inputs = block.self_attn.attn_inputs
            if attn_inputs is None:
                raise RuntimeError(
                    f"sender tracer did not capture layer {layer_idx}"
                )
            query = attn_inputs["query"]
            if self.recv_window > 0 and query.shape[-2] > self.recv_window:
                query = query[:, :, -self.recv_window :, :]
            key = out_A_past_key_values.key_cache[layer_idx]
            if query.shape[-1] != key.shape[-1]:
                raise ValueError(
                    f"sender Q/K head_dim mismatch at layer {layer_idx}: "
                    f"Q={query.shape[-1]}, K={key.shape[-1]}"
                )
            expected_q_heads = (
                key.shape[1] * block.self_attn.num_key_value_groups
            )
            if query.shape[1] != expected_q_heads:
                raise ValueError(
                    f"sender Q heads={query.shape[1]}, expected={expected_q_heads} "
                    f"at layer {layer_idx}"
                )
            weights = eager_attention_forward_without_value(
                block.self_attn,
                query=query.to(key.dtype),
                key=key,
                attention_mask=None,
                scaling=block.self_attn.scaling,
            )
            per_layer_importance[layer_idx] = (
                weights.float().mean(dim=1)[0].sum(dim=0).contiguous()
            )
        self.last_protocol_timing["t_sender_score"] = float(
            time.perf_counter() - score_start
        )
        self.last_query_sketch_cost = {
            "protocol_version": self.protocol_version,
            "query_sketch_mode": "none_sender_context_q",
            "query_sketch_tokens": 0,
            "query_sketch_layers": 0,
            "query_sketch_elements": 0,
            "query_sketch_bytes": 0,
            "query_sketch_scale_bytes": 0,
            "query_sketch_metadata_bytes": 0,
        }
        budget_start = time.perf_counter()
        self.token_importance = self._aggregate_receiver_importance(
            per_layer_importance
        )
        self._compute_budget()
        self.last_protocol_timing["t_budget_compute"] = float(
            time.perf_counter() - budget_start
        )

    @torch.no_grad()
    def compute_oracle_receiver_importance(self, input_ids_B, out_A_past_key_values):
        """Full-KV oracle receiver-aware token scoring (upper bound).

        Run B's prefill over its question with A's FULL (uncompressed) KV and measure
        how much B's query attends to each A-context token. The result drives which
        A-tokens to keep when compressing (Pass 2). This is the cross-model analogue of
        SnapKV: the *receiver* B's queries select the *sender* A's cache.

        Stores self.token_importance[A-layer i] = [L_i] importance over A's context.
        Requires apply_attn_tracer=True and (for now) same #layers for A and B.
        """
        assert self.apply_attn_tracer, "receiver scoring needs apply_attn_tracer=True"
        assert self.A_num_layers == self.B_num_layers, "receiver scoring currently supports same-depth A/B"
        self.last_protocol_timing = {}

        self._sync_tensor_device(input_ids_B)
        oracle_copy_start = time.perf_counter()
        kv_copy = copy.deepcopy(out_A_past_key_values)  # B's prefill mutates the cache -> use a copy
        ctx_len = [kv_copy.key_cache[i].shape[-2] for i in range(len(kv_copy.key_cache))]
        self._sync_tensor_device(input_ids_B)
        self.last_protocol_timing["t_oracle_kv_copy"] = float(
            time.perf_counter() - oracle_copy_start
        )

        # one extra parallel prefill of the (short) question -> tracer captures per-layer Q/K
        self._sync_tensor_device(input_ids_B)
        query_prefill_start = time.perf_counter()
        _ = self.B(input_ids=input_ids_B, past_key_values=kv_copy, use_cache=True)
        self._sync_tensor_device(input_ids_B)
        self.last_protocol_timing["t_b_query_prefill"] = float(
            time.perf_counter() - query_prefill_start
        )

        layers = self._receiver_layers()

        per_layer_importance = {}
        self._sync_tensor_device(input_ids_B)
        sender_score_start = time.perf_counter()
        for bl, block in enumerate(layers):
            attn_inputs = block.self_attn.attn_inputs
            attn_weights = eager_attention_forward_without_value(block.self_attn, **attn_inputs)  # [B,H,q,kv]
            aw = attn_weights.float().mean(dim=1)[0]  # [q, kv], mean over heads
            # observation window: only the last recv_window question rows carry the query
            # intent; summing over all rows dilutes it with template/function words.
            if self.recv_window > 0 and aw.shape[0] > self.recv_window:
                aw = aw[-self.recv_window :, :]
            imp = aw.sum(dim=0)  # [kv], attention mass each kv token receives
            # B-layer bl reads A-layer bl's KV (identity layer_map for same-depth);
            # restrict to A-context columns (drop B's own appended question keys)
            per_layer_importance[bl] = imp[: ctx_len[bl]].contiguous()

        self._sync_tensor_device(input_ids_B)
        self.last_protocol_timing["t_sender_score"] = float(
            time.perf_counter() - sender_score_start
        )
        self._sync_tensor_device(input_ids_B)
        budget_start = time.perf_counter()
        self.token_importance = self._aggregate_receiver_importance(per_layer_importance)
        self.last_query_sketch_cost = {
            "protocol_version": self.protocol_version,
            "query_sketch_tokens": 0,
            "query_sketch_layers": 0,
            "query_sketch_elements": 0,
            "query_sketch_bytes": 0,
            "query_sketch_metadata_bytes": 0,
            "oracle_full_kv_bytes": int(
                sum(
                    k.numel() * k.element_size() + v.numel() * v.element_size()
                    for k, v in zip(
                        out_A_past_key_values.key_cache,
                        out_A_past_key_values.value_cache,
                    )
                )
            ),
        }

        # budget-aware allocation depends on the importance distribution just computed
        self._compute_budget()
        self._sync_tensor_device(input_ids_B)
        self.last_protocol_timing["t_budget_compute"] = float(
            time.perf_counter() - budget_start
        )

    def _aggregate_receiver_importance(self, per_layer_importance: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Aggregate receiver-attention scores across B layers for ablations.

        identity: original ReKV, each A layer uses its paired B-layer attention.
        last:     every A layer uses the final B-layer attention.
        mean:     every A layer uses the mean attention across B layers.
        lastK:    every A layer uses the mean of the last K B layers, e.g. last4.
        topK:     every A layer uses the mean of K most concentrated B layers, e.g. top4.
        """
        mode = str(getattr(self, "receiver_layer_agg", "identity") or "identity").lower()
        if mode in {"identity", "per_layer", "none"}:
            return per_layer_importance
        if not per_layer_importance:
            return per_layer_importance

        layers = sorted(per_layer_importance)
        lengths = {int(per_layer_importance[l].numel()) for l in layers}
        if len(lengths) != 1:
            logging.warning(
                "[receiver_layer_agg] mode=%s requires equal context lengths, got %s; falling back to identity",
                mode,
                sorted(lengths),
            )
            return per_layer_importance

        if mode == "last":
            selected = [layers[-1]]
        elif mode == "mean":
            selected = layers
        elif mode.startswith("last"):
            k = int(mode[4:] or "4")
            selected = layers[-max(1, min(k, len(layers))):]
        elif mode.startswith("top"):
            k = int(mode[3:] or "4")
            scored = []
            for l in layers:
                s = per_layer_importance[l].float().clamp_min(0)
                kk = max(1, int(round(0.1 * s.numel())))
                scored.append((float(torch.topk(s, min(kk, s.numel())).values.mean()), l))
            selected = [l for _, l in sorted(scored, reverse=True)[: max(1, min(k, len(scored)))]]
        else:
            raise ValueError(f"unknown receiver_layer_agg={self.receiver_layer_agg}")

        agg = torch.stack([per_layer_importance[l].float() for l in selected], dim=0).mean(dim=0).contiguous()
        return {l: agg.clone() for l in layers}

    @torch.no_grad()
    def compute_context_attention(self, input_ids_B, compressed_past_key_values):
        """KV-sufficiency signal for progressive communication.

        Run B's question prefill over the *already-compressed* A KV and measure
        how B's query attends to the surviving A-context tokens:
          - ctx_mass: fraction of B-query attention mass landing on A-context
                      (vs B's own question tokens). Low mass => B can't ground its
                      query in the transmitted KV (starved) => likely needs more.
          - ctx_conc: concentration (1 - normalized entropy) of that attention over
                      A-context. High => B locked onto focused evidence => likely enough.
        Both aggregated over the last recv_window query rows, mean over heads/layers.
        """
        assert self.apply_attn_tracer, "context-attention signal needs apply_attn_tracer=True"
        kv_copy = copy.deepcopy(compressed_past_key_values)
        ctx_len = [kv_copy.key_cache[i].shape[-2] for i in range(len(kv_copy.key_cache))]
        _ = self.B(input_ids=input_ids_B, past_key_values=kv_copy, use_cache=True)

        if hasattr(self.B.model, "language_model"):
            layers = self.B.model.language_model.layers
        else:
            layers = self.B.model.layers

        masses, concs = [], []
        for bl, block in enumerate(layers):
            attn_inputs = block.self_attn.attn_inputs
            aw = eager_attention_forward_without_value(block.self_attn, **attn_inputs).float().mean(dim=1)[0]  # [q, kv]
            if self.recv_window > 0 and aw.shape[0] > self.recv_window:
                aw = aw[-self.recv_window:, :]
            c = ctx_len[bl]
            if c <= 0:
                continue
            ctx_aw = aw[:, :c]  # attention onto A-context columns
            total = aw.sum(dim=-1).clamp_min(1e-9)  # [q]
            masses.append((ctx_aw.sum(dim=-1) / total).mean())
            p = ctx_aw / ctx_aw.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # [q, c]
            ent = -(p * (p + 1e-12).log()).sum(dim=-1)  # [q]
            concs.append((1 - ent / math.log(max(c, 2))).mean())
        return {
            "ctx_mass": float(torch.stack(masses).mean()) if masses else 0.0,
            "ctx_conc": float(torch.stack(concs).mean()) if concs else 0.0,
        }

    @torch.no_grad()
    def compute_pass1_features(self):
        """Single-shot budget-prediction features from the receiver-importance
        distribution already computed in compute_receiver_importance (Pass-1, no
        generation). All features are dimensionless / relative so a predictor can
        transfer across tasks. Aggregated as mean & std over layers.

        The key features are `rcapXX` = fraction of A-context tokens (ranked by
        importance) needed to cover XX% of the receiver's attention mass: a direct,
        query-specific proxy for "how much KV budget this question needs".
        """
        assert self.token_importance, "call compute_receiver_importance first"
        per_layer = {k: [] for k in
                     ("rcap50", "rcap90", "rcap95", "ent", "gini", "top10", "top20", "recency", "sink")}
        L_last = 0
        for l, s in self.token_importance.items():
            s = s.float().clamp_min(0)
            L = s.numel()
            if L < 2 or float(s.sum()) <= 0:
                continue
            L_last = L
            p = s / s.sum()
            sp, _ = torch.sort(p, descending=True)
            csum = torch.cumsum(sp, dim=0)
            for thr, key in ((0.5, "rcap50"), (0.9, "rcap90"), (0.95, "rcap95")):
                idx = int(torch.searchsorted(csum, torch.tensor(thr, device=csum.device)))
                per_layer[key].append((idx + 1) / L)
            ent = -(p * (p + 1e-12).log()).sum()
            per_layer["ent"].append(float(ent / math.log(L)))
            # Gini of the importance distribution
            sp_asc = torch.sort(p, descending=False).values
            ranks = torch.arange(1, L + 1, device=p.device, dtype=torch.float)
            per_layer["gini"].append(float((2 * (ranks * sp_asc).sum()) / (L * sp_asc.sum()) - (L + 1) / L))
            per_layer["top10"].append(float(csum[max(0, int(0.10 * L) - 1)]))
            per_layer["top20"].append(float(csum[max(0, int(0.20 * L) - 1)]))
            per_layer["recency"].append(float(p[int(0.9 * L):].sum()))
            per_layer["sink"].append(float(p[: min(4, L)].sum()))

        feat = {}
        for k, v in per_layer.items():
            t = torch.tensor(v, dtype=torch.float) if v else torch.zeros(1)
            feat[f"{k}_mean"] = float(t.mean())
            feat[f"{k}_std"] = float(t.std(unbiased=False))
        feat["log_ctx_len"] = float(math.log10(max(L_last, 1)))
        return feat

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        out_A_past_key_values: Optional[Cache] = None,
        **kwargs
    ):

        if out_A_past_key_values is None:
            raise NotImplementedError("out_A_past_key_values is required when input_ids.shape[-1] > 1")
        else:
            if input_ids.shape[-1] > 1:
                out_A_past_key_values = self.prepare_key_cache(out_A_past_key_values)
            else:
                out_A_past_key_values = past_key_values
                assert past_key_values is not None, "past_key_values is required when input_ids.shape[-1] == 1"
        
        if self.shift_back:
            if type(self.B.model) == LlamaModel:
                out_B = forward_shift_back_llama(
                    model=self.B,
                    input_ids=input_ids,
                    past_key_values=out_A_past_key_values,
                    **kwargs
                )
            elif type(self.B.model) == Qwen2Model:
                out_B = forward_shift_back_qwen2(
                    model=self.B,
                    input_ids=input_ids,
                    past_key_values=out_A_past_key_values,
                    **kwargs
                )
            else:
                raise NotImplementedError(f"shift_back is not implemented for model type {type(self.B)}")
        else:
            out_B = self.B(
                input_ids=input_ids,
                past_key_values=out_A_past_key_values,
                **kwargs
            )

        return out_B

    @torch.no_grad()
    def calc_attn_weights_from_qk(self):
        assert self.apply_attn_tracer, "apply_attn_tracer must be True"
        if hasattr(self.B.model, "language_model"):
            layers = self.B.model.language_model.layers
        else:
            layers = self.B.model.layers
        for i, block in enumerate(layers):
            attn_inputs = block.self_attn.attn_inputs
            attn_weights = eager_attention_forward_without_value(block.self_attn, **attn_inputs)
            # attn_weights_sdpa = sdpa_attention_forward_without_value(block.self_attn, **attn_inputs)
            self.B_attn_weights[i] = attn_weights

def eager_attention_forward_without_value(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    return attn_weights


def sdpa_attention_forward_without_value(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> torch.Tensor:

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    eye = torch.eye(key.shape[-2], dtype=key.dtype, device=key.device)
    value_eye = eye.unsqueeze(0).unsqueeze(0).expand(key.shape[0], key.shape[1], -1, -1)

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value_eye,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_weights = attn_output

    return attn_weights

import copy
def get_short_past_key_values(past_key_values: DynamicCache):
    lengths = set()
    for idx in range(len(past_key_values.key_cache)):
        if past_key_values.key_cache[idx].numel():
            lengths.add(past_key_values.key_cache[idx].shape[-2])
    assert len(lengths) <= 2
    short_past_key_values = copy.deepcopy(past_key_values)
    short_past_key_values.crop(min(lengths))
    short_length = min(lengths)
    return short_past_key_values, short_length


from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
def forward_shift_back_llama(
    model: PreTrainedModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if past_key_values is None:
        past_key_values = DynamicCache()
    
    ##########
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )

    position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    ##########
    ##########
    short_past_key_values, short_length = get_short_past_key_values(past_key_values)
    past_seen_tokens = short_past_key_values.get_seq_length() if short_past_key_values is not None else 0
    short_cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )

    short_position_ids = short_cache_position.unsqueeze(0)

    short_causal_mask = create_causal_mask(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=short_cache_position,
        past_key_values=short_past_key_values,
        position_ids=short_position_ids,
    )
    ##########
    # print("short_length:", short_length)
    # # print("causal_mask shape:", causal_mask.shape)
    # # print("short_causal_mask shape:", short_causal_mask.shape)
    # print("position_ids:", position_ids)
    # print("short_position_ids:", short_position_ids)
    # print("cache_position:", cache_position)
    # print("short_cache_position:", short_cache_position)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    short_position_embeddings = model.model.rotary_emb(hidden_states, short_position_ids)
    
    all_hidden_states = ()

    for i, decoder_layer in enumerate(model.model.layers[: model.config.num_hidden_layers]):

        all_hidden_states += (hidden_states,)

        if past_key_values.key_cache[i].shape[-2] == short_length:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=short_causal_mask,
                position_ids=short_position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=short_cache_position,
                position_embeddings=short_position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

    hidden_states = model.model.norm(hidden_states)

    all_hidden_states += (hidden_states,)

    # Causal LM
    logits_to_keep = 0
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = model.lm_head(hidden_states[:, slice_indices, :])


    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=None,
    )

def forward_shift_back_qwen2(
    model: PreTrainedModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if past_key_values is None:
        past_key_values = DynamicCache()
    
    ##########
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )

    position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": model.model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if model.model.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    ##########
    ##########
    short_past_key_values, short_length = get_short_past_key_values(past_key_values)
    past_seen_tokens = short_past_key_values.get_seq_length() if short_past_key_values is not None else 0
    short_cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )

    short_position_ids = short_cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(short_causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": model.model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": short_cache_position,
            "past_key_values": short_past_key_values,
            "position_ids": short_position_ids,
        }
        # Create the masks
        short_causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if model.model.has_sliding_layers:
            short_causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    ##########
    # print("short_length:", short_length)
    # # print("causal_mask shape:", causal_mask.shape)
    # # print("short_causal_mask shape:", short_causal_mask.shape)
    # print("position_ids:", position_ids)
    # print("short_position_ids:", short_position_ids)
    # print("cache_position:", cache_position)
    # print("short_cache_position:", short_cache_position)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    short_position_embeddings = model.model.rotary_emb(hidden_states, short_position_ids)
    
    all_hidden_states = ()

    for i, decoder_layer in enumerate(model.model.layers[: model.config.num_hidden_layers]):

        all_hidden_states += (hidden_states,)

        if past_key_values.key_cache[i].shape[-2] == short_length:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=short_causal_mask_mapping[decoder_layer.attention_type],
                position_ids=short_position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=short_cache_position,
                position_embeddings=short_position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=model.model.config.output_attentions,
                use_cache=model.model.config.use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

    hidden_states = model.model.norm(hidden_states)

    all_hidden_states += (hidden_states,)

    # Causal LM
    logits_to_keep = 0
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = model.lm_head(hidden_states[:, slice_indices, :])


    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=None,
    )

