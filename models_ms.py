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
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention


class CVCommunicator(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_A1: PreTrainedModel,
        model_A2: PreTrainedModel,
        model_B: PreTrainedModel,
        layer_from: int,
        layer_to: int,
        top_layers: float = 0.0,
        layers_list: list[int] = [],
        apply_attn_tracer: bool = False,
        merge: bool = False,
        merge_ratio: float = 0.2,
        merge_sink: int = 4,
        merge_recent: int = 8,
        merge_mode: str = "merge",
        score_mode: str = "value_norm",
        recv_window: int = 0,
        query_sketch_mode: str = "bf16",
        budget_mode: str = "uniform",
        budget_min: float = 0.05,
        budget_max: float = 0.5,
        budget_tau: float = 1.0,
        budget_floor: float = 0.02,
        coverage_tau: float = 0.90,
        coverage_scale: float = 1.0,
    ) -> None:
        super().__init__(model_B.config)
        self.A1 = model_A1
        self.A2 = model_A2
        self.B = model_B
        self.layer_from = layer_from
        self.layer_to = layer_to
        self.apply_attn_tracer = apply_attn_tracer
        self.merge = merge
        self.merge_ratio = merge_ratio
        self.merge_sink = merge_sink
        self.merge_recent = merge_recent
        self.merge_mode = merge_mode
        self.score_mode = score_mode
        self.recv_window = recv_window
        self.query_sketch_mode = str(query_sketch_mode).lower()
        if self.query_sketch_mode != "bf16":
            raise ValueError("multi-source Query-Sketch currently supports query_sketch_mode=bf16 only")
        self.budget_mode = budget_mode
        self.budget_min = budget_min
        self.budget_max = budget_max
        self.budget_tau = budget_tau
        self.budget_floor = budget_floor
        self.coverage_tau = coverage_tau
        self.coverage_scale = coverage_scale
        self.token_importance = None
        self.layer_budget = {}
        self.last_query_budget = None
        self.last_kept_ratio = None
        self.last_kv_cost = None
        self.last_query_sketch_cost = {}
        self.last_protocol_timing = {}
        self.protocol_version = self._protocol_version()
        self._merge_logged = False
        for p in self.A1.parameters(): p.requires_grad = False
        for p in self.A2.parameters(): p.requires_grad = False
        for p in self.B.parameters(): p.requires_grad = False

        if hasattr(self.A1.config, "num_hidden_layers"):
            self.A1_num_layers = self.A1.config.num_hidden_layers
        elif hasattr(self.A1.config, "text_config") and hasattr(self.A1.config.text_config, "num_hidden_layers"):
            self.A1_num_layers = self.A1.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.A1.config}")
        if hasattr(self.A2.config, "num_hidden_layers"):
            self.A2_num_layers = self.A2.config.num_hidden_layers
        elif hasattr(self.A2.config, "text_config") and hasattr(self.A2.config.text_config, "num_hidden_layers"):
            self.A2_num_layers = self.A2.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.A2.config}")
        if hasattr(self.B.config, "num_hidden_layers"):
            self.B_num_layers = self.B.config.num_hidden_layers
        elif hasattr(self.B.config, "text_config") and hasattr(self.B.config.text_config, "num_hidden_layers"):
            self.B_num_layers = self.B.config.text_config.num_hidden_layers
        else:
            raise ValueError(f"num_hidden_layers not found in {self.B.config}")

        assert self.A1_num_layers == self.A2_num_layers == self.B_num_layers, "model_A1, model_A2 and model_B must have the same number of layers"

        if layers_list[0] != -1:
            self.layers_list = layers_list
        elif top_layers > 0:
            self.layers_list = list(range(0, self.A1_num_layers)) # set all layers at first
        else:
            self.layers_list = list(range(self.layer_from, self.layer_to + 1))
        if apply_attn_tracer:
            self.B_attn_weights = {}
            self.apply_B_attn_tracer()

        logging.info(f"CVCommunicator initialized")

    def _protocol_version(self) -> str:
        if self.score_mode == "receiver_oracle":
            return "full_kv_oracle_multi_source_v1"
        if self.score_mode == "receiver":
            return f"query_sketch_{self.query_sketch_mode}_multi_source_v1"
        return "query_agnostic_kv_multi_source_v1"

    @staticmethod
    def _sync_tensor_device(tensor: torch.Tensor):
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def apply_B_attn_tracer(self):
        if hasattr(self.B.model, "language_model"):
            layers = self.B.model.language_model.layers
        else:
            layers = self.B.model.layers
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

    def _append_sender_layer(self, past_key_values_new, key, value, layer_idx, sender_idx):
        if self.merge:
            if layer_idx == 0:
                key_i, value_i = key, value
            else:
                key_i, value_i = self.compress_merge_layer(key, value, layer_idx=layer_idx, sender_idx=sender_idx)
        elif layer_idx in self.layers_list or layer_idx == 0:
            key_i, value_i = key, value
        else:
            # keep the first token due to attention sink
            key_i = key[:, :, :1, :]
            value_i = value[:, :, :1, :]
        past_key_values_new.update(key_i, value_i, layer_idx)
        return key_i, value_i

    def prepare_key_cache(self, A1_past_key_values, A2_past_key_values, compress: bool = True):
        A1_key_cache = A1_past_key_values.key_cache
        A1_value_cache = A1_past_key_values.value_cache
        A2_key_cache = A2_past_key_values.key_cache
        A2_value_cache = A2_past_key_values.value_cache
        past_key_values_new = DynamicCache()
        kept_tokens, total_tokens = 0, 0
        kept_kv_elements, total_kv_elements = 0, 0
        kept_kv_bytes, total_kv_bytes = 0, 0
        for i in range(len(A1_key_cache)): # i is the layer index of model A
            for key, value in ((A1_key_cache[i], A1_value_cache[i]), (A2_key_cache[i], A2_value_cache[i])):
                total_tokens += key.shape[-2]
                total_kv_elements += key.numel() + value.numel()
                total_kv_bytes += key.numel() * key.element_size() + value.numel() * value.element_size()

            if compress:
                key_cache_i, value_cache_i = self._append_sender_layer(
                    past_key_values_new, A1_key_cache[i], A1_value_cache[i], i, sender_idx=0
                )
                key_cache_j, value_cache_j = self._append_sender_layer(
                    past_key_values_new, A2_key_cache[i], A2_value_cache[i], i, sender_idx=1
                )
            else:
                key_cache_i, value_cache_i = A1_key_cache[i], A1_value_cache[i]
                key_cache_j, value_cache_j = A2_key_cache[i], A2_value_cache[i]
                past_key_values_new.update(key_cache_i, value_cache_i, i)
                past_key_values_new.update(key_cache_j, value_cache_j, i)

            for key, value in ((key_cache_i, value_cache_i), (key_cache_j, value_cache_j)):
                kept_tokens += key.shape[-2]
                kept_kv_elements += key.numel() + value.numel()
                kept_kv_bytes += key.numel() * key.element_size() + value.numel() * value.element_size()
        self.last_kept_ratio = kept_tokens / total_tokens if total_tokens else None
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
        index_bytes = int(kept_tokens * 4)
        kv_metadata_bytes = int(8 + len(A1_key_cache) * 20)
        sketch_bytes = int(self.last_query_sketch_cost.get("query_sketch_bytes", 0))
        sketch_metadata_bytes = int(self.last_query_sketch_cost.get("query_sketch_metadata_bytes", 0))
        # The receiver sends the same sketch to both independent senders.
        b_to_a_bytes = 2 * (sketch_bytes + sketch_metadata_bytes)
        a_to_b_bytes = int(kept_kv_bytes + index_bytes + kv_metadata_bytes)
        self.last_kv_cost.update({
            "selection_index_bytes": index_bytes,
            "kv_metadata_bytes": kv_metadata_bytes,
            "a_to_b_communication_bytes": a_to_b_bytes,
            "b_to_a_communication_bytes": b_to_a_bytes,
            "total_communication_bytes": a_to_b_bytes + b_to_a_bytes,
        })
        return past_key_values_new

    @torch.no_grad()
    def compress_merge_layer(self, K: torch.Tensor, V: torch.Tensor, layer_idx: int = -1, sender_idx: int = 0):
        """Compress one sender's KV at one layer before appending it to B's cache."""
        B, H, L, D = K.shape
        r_eff = self._effective_ratio(layer_idx)
        k = max(int(round(r_eff * L)), self.merge_sink + self.merge_recent)
        if k >= L:
            return K, V

        if (
            self.score_mode in {"receiver", "receiver_oracle"}
            and self.token_importance is not None
            and layer_idx in self.token_importance
            and len(self.token_importance[layer_idx]) > sender_idx
            and self.token_importance[layer_idx][sender_idx].numel() == L
        ):
            imp = self.token_importance[layer_idx][sender_idx].to(K.device).float().clone()
        elif self.score_mode == "random":
            imp = torch.rand(L, device=K.device, dtype=torch.float32)
        else:
            imp = V.float().norm(dim=-1).mean(dim=1)[0]

        big = torch.finfo(imp.dtype).max
        if self.merge_sink > 0:
            imp[: self.merge_sink] = big
        if self.merge_recent > 0:
            imp[L - self.merge_recent :] = big

        keep = torch.topk(imp, k).indices
        keep, _ = torch.sort(keep)
        evict_mask = torch.ones(L, dtype=torch.bool, device=K.device)
        evict_mask[keep] = False
        evict = torch.nonzero(evict_mask, as_tuple=False).squeeze(-1)

        Kk = K[:, :, keep, :].contiguous()
        Vk = V[:, :, keep, :].clone()
        if self.merge_mode == "merge" and evict.numel() > 0:
            Ke = K[:, :, evict, :]
            Ve = V[:, :, evict, :]
            sim = torch.matmul(Ke, Kk.transpose(-1, -2)) / (D ** 0.5)
            w = torch.softmax(sim.float(), dim=-1)
            agg = torch.matmul(w.transpose(-1, -2), Ve.float())
            denom = 1.0 + w.sum(dim=-2, keepdim=True).transpose(-1, -2)
            Vk = ((Vk.float() + agg) / denom).to(V.dtype)

        if not self._merge_logged:
            logging.info(
                f"[multi-source merge] mode={self.merge_mode} score={self.score_mode} "
                f"budget={self.budget_mode} layer compress: L={L} -> k={k} "
                f"(r_eff={r_eff:.3f}, base_ratio={self.merge_ratio}, sink={self.merge_sink}, recent={self.merge_recent})"
            )
            self._merge_logged = True
        return Kk, Vk.contiguous()

    def _effective_ratio(self, layer_idx: int) -> float:
        if self.budget_mode == "uniform" or not self.layer_budget:
            return self.merge_ratio
        return self.layer_budget.get(layer_idx, self.merge_ratio)

    def _combined_importance(self, layer_idx: int):
        if not self.token_importance or layer_idx not in self.token_importance:
            return None
        parts = [s.float().clamp_min(0) for s in self.token_importance[layer_idx] if s.numel() > 0]
        if not parts:
            return None
        return torch.cat(parts, dim=0)

    def _query_total_budget(self) -> float:
        Hns = []
        for layer_idx in sorted(self.token_importance.keys()):
            if layer_idx == 0:
                continue
            s = self._combined_importance(layer_idx)
            if s is None or s.numel() < 2 or s.sum() <= 0:
                continue
            p = s / (s.sum() + 1e-9)
            H = -(p * (p + 1e-12).log()).sum()
            Hns.append((H / math.log(s.numel())).clamp(0.0, 1.0).item())
        if not Hns:
            return self.merge_ratio
        Hmean = sum(Hns) / len(Hns)
        return self.budget_min + (self.budget_max - self.budget_min) * Hmean

    def _coverage_ratio(self, s: torch.Tensor) -> float:
        s = s.float().clamp_min(0)
        L = s.numel()
        if L < 2 or float(s.sum()) <= 0:
            return self.merge_ratio
        p = s / s.sum().clamp_min(1e-9)
        sp = torch.sort(p, descending=True).values
        csum = torch.cumsum(sp, dim=0)
        thr = min(max(float(self.coverage_tau), 0.0), 1.0)
        idx = int(torch.searchsorted(csum, torch.tensor(thr, device=csum.device)))
        idx = min(idx, L - 1)
        raw = ((idx + 1) / L) * float(self.coverage_scale)
        lo = max(float(self.budget_min), float(self.budget_floor))
        hi = max(float(self.budget_max), lo)
        return float(min(max(raw, lo), hi))

    def _compute_budget(self):
        self.layer_budget = {}
        self.last_query_budget = None
        if self.budget_mode == "uniform" or not self.token_importance:
            return
        comp = [l for l in sorted(self.token_importance.keys()) if l != 0]
        if not comp:
            return
        if self.budget_mode == "coverage":
            self.layer_budget = {
                l: self._coverage_ratio(self._combined_importance(l))
                for l in comp
                if self._combined_importance(l) is not None
            }
            if self.layer_budget:
                self.last_query_budget = sum(self.layer_budget.values()) / len(self.layer_budget)
            return
        B = self._query_total_budget() if self.budget_mode in ("query", "query+layer") else self.merge_ratio
        self.last_query_budget = float(B)
        if self.budget_mode in ("layer", "query+layer"):
            I = []
            layers = []
            for l in comp:
                s = self._combined_importance(l)
                if s is None or s.numel() == 0:
                    continue
                kk = max(1, int(round(0.1 * s.numel())))
                I.append(torch.topk(s, min(kk, s.numel())).values.mean())
                layers.append(l)
            if I:
                Ivec = torch.stack(I)
                w = torch.softmax(Ivec / self.budget_tau, dim=0)
                r = (w * len(layers) * B).clamp(self.budget_floor, 1.0)
                self.layer_budget = {l: float(r[i]) for i, l in enumerate(layers)}
        else:
            self.layer_budget = {l: float(B) for l in comp}

    @torch.no_grad()
    def compute_receiver_importance(self, input_ids_B, out_A1_past_key_values, out_A2_past_key_values):
        """Score both senders locally from one deployable B-to-A query sketch."""
        assert self.apply_attn_tracer, "receiver scoring needs apply_attn_tracer=True"
        self.last_protocol_timing = {}
        ctx_lens = [
            (
                out_A1_past_key_values.key_cache[i].shape[-2],
                out_A2_past_key_values.key_cache[i].shape[-2],
            )
            for i in range(self.A1_num_layers)
        ]
        if len({a + b for a, b in ctx_lens}) != 1:
            raise ValueError("multi-source query-sketch requires equal total context length across layers")
        context_length = sum(ctx_lens[0])
        query_length = int(input_ids_B.shape[-1])
        position_ids = torch.arange(
            context_length,
            context_length + query_length,
            device=input_ids_B.device,
            dtype=torch.long,
        ).unsqueeze(0)

        self._sync_tensor_device(input_ids_B)
        prefill_start = time.perf_counter()
        _ = self.B(
            input_ids=input_ids_B,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        self._sync_tensor_device(input_ids_B)
        self.last_protocol_timing["t_b_query_prefill"] = float(
            time.perf_counter() - prefill_start
        )

        layers = self.B.model.language_model.layers if hasattr(
            self.B.model, "language_model"
        ) else self.B.model.layers
        self.token_importance = {}
        sketch_elements = 0
        sketch_bytes = 0
        sketch_tokens = 0
        self._sync_tensor_device(input_ids_B)
        score_start = time.perf_counter()
        for bl, block in enumerate(layers):
            attn_inputs = block.self_attn.attn_inputs
            if attn_inputs is None:
                raise RuntimeError(f"query tracer did not capture layer {bl}")
            query = attn_inputs["query"]
            if self.recv_window > 0 and query.shape[-2] > self.recv_window:
                query = query[:, :, -self.recv_window :, :]
            wire_query = query.to(torch.bfloat16).contiguous()
            sketch_elements += wire_query.numel()
            sketch_bytes += wire_query.numel() * wire_query.element_size()
            sketch_tokens = int(wire_query.shape[-2])

            importance = []
            for key in (
                out_A1_past_key_values.key_cache[bl],
                out_A2_past_key_values.key_cache[bl],
            ):
                local_query = wire_query.to(key.dtype)
                if local_query.shape[-1] != key.shape[-1]:
                    raise ValueError(
                        f"multi-source query-sketch head_dim mismatch at layer {bl}: "
                        f"Q={local_query.shape[-1]}, K={key.shape[-1]}"
                    )
                expected_heads = key.shape[1] * block.self_attn.num_key_value_groups
                if local_query.shape[1] != expected_heads:
                    raise ValueError(
                        f"multi-source query-sketch head mismatch at layer {bl}: "
                        f"Q heads={local_query.shape[1]}, expected={expected_heads}"
                    )
                weights = eager_attention_forward_without_value(
                    block.self_attn,
                    query=local_query,
                    key=key,
                    attention_mask=None,
                    scaling=block.self_attn.scaling,
                )
                importance.append(
                    weights.float().mean(dim=1)[0].sum(dim=0).contiguous()
                )
            self.token_importance[bl] = [
                importance[0],
                importance[1],
            ]
        self._sync_tensor_device(input_ids_B)
        self.last_protocol_timing["t_sender_score"] = float(
            time.perf_counter() - score_start
        )
        self.last_query_sketch_cost = {
            "protocol_version": self.protocol_version,
            "query_sketch_mode": self.query_sketch_mode,
            "query_sketch_tokens": sketch_tokens,
            "query_sketch_layers": self.B_num_layers,
            "query_sketch_elements": int(sketch_elements),
            "query_sketch_bytes": int(sketch_bytes),
            "query_sketch_metadata_bytes": int(8 + self.B_num_layers * 16),
        }
        self._compute_budget()

    @torch.no_grad()
    def compute_oracle_receiver_importance(self, input_ids_B, out_A1_past_key_values, out_A2_past_key_values):
        """Full-KV multi-source receiver scoring retained only as an oracle."""
        assert self.apply_attn_tracer, "receiver scoring needs apply_attn_tracer=True"
        A1_copy = copy.deepcopy(out_A1_past_key_values)
        A2_copy = copy.deepcopy(out_A2_past_key_values)
        ctx_lens = [
            (A1_copy.key_cache[i].shape[-2], A2_copy.key_cache[i].shape[-2])
            for i in range(len(A1_copy.key_cache))
        ]
        kv_copy = self.prepare_key_cache(A1_copy, A2_copy, compress=False)
        _ = self.B(input_ids=input_ids_B, past_key_values=kv_copy, use_cache=True)
        layers = self.B.model.language_model.layers if hasattr(
            self.B.model, "language_model"
        ) else self.B.model.layers
        self.token_importance = {}
        for bl, block in enumerate(layers):
            attn_weights = eager_attention_forward_without_value(
                block.self_attn, **block.self_attn.attn_inputs
            )
            aw = attn_weights.float().mean(dim=1)[0]
            if self.recv_window > 0 and aw.shape[0] > self.recv_window:
                aw = aw[-self.recv_window :, :]
            imp = aw.sum(dim=0)
            len_a1, len_a2 = ctx_lens[bl]
            self.token_importance[bl] = [
                imp[:len_a1].contiguous(),
                imp[len_a1 : len_a1 + len_a2].contiguous(),
            ]
        self.last_query_sketch_cost = {
            "protocol_version": self.protocol_version,
            "query_sketch_bytes": 0,
            "query_sketch_metadata_bytes": 0,
            "oracle_full_kv_bytes": int(
                sum(
                    key.numel() * key.element_size() + value.numel() * value.element_size()
                    for cache in (out_A1_past_key_values, out_A2_past_key_values)
                    for key, value in zip(cache.key_cache, cache.value_cache)
                )
            ),
        }
        self._compute_budget()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        out_A1_past_key_values: Optional[Cache] = None,
        out_A2_past_key_values: Optional[Cache] = None,
        **kwargs
    ):
        if out_A1_past_key_values is None or out_A2_past_key_values is None:
            raise NotImplementedError("out_A1_past_key_values and out_A2_past_key_values are required when input_ids.shape[-1] > 1")
        else:
            if input_ids.shape[-1] > 1:
                past_key_values = self.prepare_key_cache(out_A1_past_key_values, out_A2_past_key_values)
            else:
                assert past_key_values is not None, "past_key_values is required when input_ids.shape[-1] == 1"
        out_B = self.B(
            input_ids=input_ids,
            past_key_values=past_key_values,
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
