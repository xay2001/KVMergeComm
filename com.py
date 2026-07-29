import os
import torch
import argparse
import wandb
import datetime
import logging
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed
from models_ac import ActivationCommunicator
from models import CVCommunicator
from models_cipher import CipherAgent
from typing import Literal
from utils import setup_logging, log_gpu_info, generate_run_name
from dataloader import get_evaluator
from eval import (
    SkylineEvaluator,
    CommunicationEvaluator,
    BaselineEvaluator,
    ACEvaluator,
    NLDEvaluator,
    QueryAwareTextRetrievalEvaluator,
    CipherEvaluator,
)
from layer_importance import get_top_layers, get_layer_ranking
import random

RECEIVER_AWARE_SCORE_MODES = {
    "receiver",
    "receiver_oracle",
    "receiver_x_value_norm",
    "receiver_value_norm",
    "receiver_recency",
    "receiver_recency_prior",
}

@dataclass
class AlignConfig:
    # device configuration
    device: str = "cuda:0"
    seed: int = 42
    snapshot_path: str = "snapshots"
    # model configuration
    model_A: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_B: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_input_length: int = 64 * 1000
    # Communication configuration
    layer_from: int = 0
    layer_to: int = 26
    layers_list: list[int] = field(default_factory=lambda: [-1])
    top_layers: float = 0.0
    calib_size: int = 1
    do_layer_curve: bool = False
    alpha: float = 1.0
    mu: float = 0.5
    sigma: float = 10.0
    random_selection: bool = False
    shift_back: bool = False
    # Merge-then-Communicate (MtC) configuration
    merge: bool = False
    merge_ratio: float = 0.2
    merge_sink: int = 4
    merge_recent: int = 8
    merge_mode: str = "merge"  # "merge" (normalized value merge) or "evict" (drop only)
    score_mode: str = "value_norm"  # "receiver" = deployable Q sketch; "receiver_oracle" = full-A-KV upper bound
    recv_window: int = 0  # Q-sketch tokens per layer: 0 = all query tokens, >0 = last N
    query_sketch_mode: str = "bf16"  # bf16 | int8 | token_ids
    receiver_layer_agg: str = "identity"  # receiver scoring layer aggregation: identity | last | mean | topK | lastK
    # Receiver-aware LAYER selection (layer granularity, receiver-conditioned):
    # per-sample, aggregate the ReKV query-sketch token importance into layer
    # scores and transmit only the top fraction of layers in full. Requires a
    # receiver-aware score_mode and merge=False. 0 disables.
    receiver_layer_fraction: float = 0.0
    receiver_layer_score: str = "topk_share"  # topk_share | entropy | max
    # Receiver-conditioning causal controls. Scoring may use a different query
    # from generation, while generation always receives the correct prompt_B.
    query_condition_mode: str = "correct"
    query_condition_seed: int = 42
    query_unrelated_task: str = ""
    budget_replay_from: str = ""
    budget_replay_mode: str = "aligned"  # aligned | shuffled
    budget_replay_seed: int = 42
    budget_replay_tolerance: float = 1e-3
    # budget-aware allocation (Step 1): uniform | query | layer | query+layer
    budget_mode: str = "uniform"
    budget_min: float = 0.05  # query-adaptive total budget lower bound
    budget_max: float = 0.5   # query-adaptive total budget upper bound
    budget_tau: float = 1.0   # softmax temperature for layer allocation
    budget_floor: float = 0.02  # per-layer minimum keep ratio
    coverage_tau: float = 0.90   # coverage budget: receiver-attention mass target
    coverage_scale: float = 1.0  # coverage budget: multiplier applied to rcap before clamp
    coverage_tau_mode: str = "fixed"  # strict_coverage: fixed | adaptive
    coverage_tau_min: float = 0.80  # adaptive strict-coverage lower target
    coverage_tau_max: float = 0.95  # adaptive strict-coverage upper target
    # Step 2b: online progressive communication
    progressive: bool = False
    prog_ladder: str = "0.1,0.2,0.3,0.5"  # ascending budget rungs for the progressive sweep
    # 牌2: single-shot budget-prediction feature dump (Pass-1 only, no generation)
    dump_pass1_features: bool = False
    # Cost profiling: run a small controlled subset and dump timing/payload stats
    # next to the run log instead of the normal per-sample eval file.
    profile_cost: bool = False
    profile_limit: int = 50
    profile_warmup: int = 5
    # Test dataset configuration
    test_task: str = "tipsheets"
    task_name: str = ""
    limit: int = 0
    # Test configuration
    do_test: bool = False
    do_test_skyline: bool = False
    do_test_baseline: bool = False
    do_test_ac: bool = False
    do_test_nld: bool = False
    do_test_text_retrieval: bool = False
    do_test_cipher: bool = False
    # NLD configuration
    # max tokens to generate for model A and B in phase 1
    nld_max_tokens_model_A_and_B_phase1: int = 128
    sender_aware: bool = False
    # receiver-aware NLD: give the sender the receiver's query text (fair-text
    # baseline), instead of the default query-blind NLD protocol
    nld_receiver_aware: bool = False
    # Training-free receiver-aware original-text retrieval baseline.
    text_retrieval_top_k: int = 4
    text_retrieval_chunk_tokens: int = 128
    text_retrieval_chunk_stride: int = 96
    text_retrieval_bm25_k1: float = 1.5
    text_retrieval_bm25_b: float = 0.75
    # AC configuration
    f: Literal["replace", "sum", "mean"] = "replace"
    layer_k: int = 26
    layer_j: int = 26
    # W&B configuration
    run_name: str = ""
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_tags: str = ""  # comma-separated tags
    # Logging configuration
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

def main(cfg: AlignConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.snapshot_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = generate_run_name(cfg) if cfg.run_name == "" else cfg.run_name
    run_name = f"{run_name}_{timestamp}"

    final_snapshot_path = os.path.join(cfg.snapshot_path, run_name)
    os.makedirs(final_snapshot_path, exist_ok=True)
    log_file_path = os.path.join(final_snapshot_path, "log.log")

    setup_logging(log_file_path=log_file_path, log_level=cfg.log_level)
    logging.info(f"Configuration: {cfg}")
    logging.info(f"All files (logs, models, metrics) will be saved to: {final_snapshot_path}")
    logging.info(f"Log level: {cfg.log_level}")
    log_gpu_info()

    # Initialize W&B
    if cfg.use_wandb:
        wandb_config = {
            k: v for k, v in cfg.__dict__.items() 
            if not k.startswith('wandb_')
        }
        
        wandb_tags = []
        if cfg.wandb_tags != "":
            wandb_tags = [tag.strip() for tag in cfg.wandb_tags.split(',')]
        
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            entity=cfg.wandb_entity,
            tags=wandb_tags,
            config=wandb_config
        )

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_B)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    needs_model_A = any(
        [
            cfg.do_test_skyline,
            cfg.do_test_baseline,
            cfg.do_test,
            cfg.do_test_ac,
            cfg.do_test_nld,
            cfg.do_test_cipher,
        ]
    )
    model_A = None
    if needs_model_A:
        model_A = AutoModelForCausalLM.from_pretrained(
            cfg.model_A,
            device_map={"": cfg.device},
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model_B = AutoModelForCausalLM.from_pretrained(cfg.model_B, device_map={"": cfg.device}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if model_A is not None:
        model_A.eval()
    model_B.eval()

    # special case for Gemma
    if "gemma" in cfg.model_A.lower() or "gemma" in cfg.model_B.lower():
        torch._dynamo.config.cache_size_limit = 64

    if model_A is not None:
        model_A.name = cfg.model_A
    model_B.name = cfg.model_B

    evaluator = get_evaluator(cfg.test_task)
    causal_query_modes = {
        "shuffled",
        "unrelated",
        "sender_text_receiver_encoder",
        "sender_context_q",
    }
    if (
        cfg.query_condition_mode in causal_query_modes
        and cfg.score_mode not in RECEIVER_AWARE_SCORE_MODES
    ):
        raise ValueError(
            f"query_condition_mode={cfg.query_condition_mode} requires a "
            "receiver-aware score_mode"
        )
    if (
        cfg.query_condition_mode == "query_free"
        and cfg.score_mode in RECEIVER_AWARE_SCORE_MODES
    ):
        raise ValueError("query_free conditioning requires a query-agnostic score_mode")
    if cfg.budget_replay_from and cfg.budget_mode != "uniform":
        raise ValueError("budget replay requires --budget_mode uniform")
    if cfg.budget_replay_mode not in {"aligned", "shuffled"}:
        raise ValueError(
            "--budget_replay_mode must be one of: aligned, shuffled"
        )
    if cfg.budget_replay_mode == "shuffled" and not cfg.budget_replay_from:
        raise ValueError("shuffled budget replay requires --budget_replay_from")
    unrelated_evaluator = None
    if cfg.query_condition_mode == "unrelated":
        if not cfg.query_unrelated_task:
            raise ValueError(
                "--query_unrelated_task is required for unrelated conditioning"
            )
        if cfg.query_unrelated_task == cfg.test_task:
            raise ValueError("unrelated conditioning task must differ from test_task")
        unrelated_evaluator = get_evaluator(cfg.query_unrelated_task)
    if cfg.budget_replay_from and not os.path.isfile(cfg.budget_replay_from):
        raise ValueError(
            f"budget replay file does not exist: {cfg.budget_replay_from}"
        )
    
    if cfg.limit == 0:
        cfg.limit = None

    results = None
    if cfg.do_test_skyline:
        skyline_evaluator = SkylineEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length)
        results = skyline_evaluator.test(model_A, model_B, limit=cfg.limit)
    if cfg.do_test_baseline:
        baseline_evaluator = BaselineEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length)
        results = baseline_evaluator.test(model_A, model_B, limit=cfg.limit)
    if cfg.do_test:
        communication_evaluator = CommunicationEvaluator(
            evaluator,
            tokenizer,
            cfg.use_wandb,
            cfg.max_input_length,
            query_condition_mode=cfg.query_condition_mode,
            query_condition_seed=cfg.query_condition_seed,
            unrelated_evaluator=unrelated_evaluator,
            budget_replay_from=cfg.budget_replay_from,
            budget_replay_mode=cfg.budget_replay_mode,
            budget_replay_seed=cfg.budget_replay_seed,
        )
        if cfg.merge:
            # Merge-then-Communicate: keep all layers, compress tokens within each via merging
            cv = CVCommunicator(
                model_A,
                model_B,
                cfg.layer_from,
                cfg.layer_to,
                layers_list=cfg.layers_list,
                top_layers=cfg.top_layers,
                apply_attn_tracer=(cfg.score_mode in RECEIVER_AWARE_SCORE_MODES),
                shift_back=cfg.shift_back,
                merge=True,
                merge_ratio=cfg.merge_ratio,
                merge_sink=cfg.merge_sink,
                merge_recent=cfg.merge_recent,
                merge_mode=cfg.merge_mode,
                score_mode=cfg.score_mode,
                recv_window=cfg.recv_window,
                query_sketch_mode=cfg.query_sketch_mode,
                receiver_layer_agg=cfg.receiver_layer_agg,
                budget_mode=cfg.budget_mode,
                budget_min=cfg.budget_min,
                budget_max=cfg.budget_max,
                budget_tau=cfg.budget_tau,
                budget_floor=cfg.budget_floor,
                coverage_tau=cfg.coverage_tau,
                coverage_scale=cfg.coverage_scale,
                coverage_tau_mode=cfg.coverage_tau_mode,
                coverage_tau_min=cfg.coverage_tau_min,
                coverage_tau_max=cfg.coverage_tau_max,
                query_condition_mode=cfg.query_condition_mode,
                budget_replay_tolerance=cfg.budget_replay_tolerance,
            ).to(cfg.device)
            if cfg.dump_pass1_features:
                communication_evaluator.dump_pass1_features(model_A, cv, limit=cfg.limit)
                results = None
            elif cfg.progressive:
                ladder = [float(x) for x in cfg.prog_ladder.split(",")]
                communication_evaluator.test_progressive(model_A, cv, ladder, limit=cfg.limit)
                results = None
            elif cfg.profile_cost:
                results = communication_evaluator.test_cost_profile(model_A, cv, limit=cfg.profile_limit, warmup=cfg.profile_warmup)
            else:
                results = communication_evaluator.test(model_A, cv, limit=cfg.limit)
        elif cfg.receiver_layer_fraction > 0:
            # Receiver-aware layer selection: same query sketch as ReKV, but the
            # importance is aggregated per layer and whole layers are transmitted.
            if cfg.score_mode not in RECEIVER_AWARE_SCORE_MODES:
                raise ValueError(
                    "--receiver_layer_fraction requires a receiver-aware score_mode"
                )
            cv = CVCommunicator(
                model_A,
                model_B,
                cfg.layer_from,
                cfg.layer_to,
                layers_list=cfg.layers_list,
                top_layers=0.0,
                apply_attn_tracer=True,
                shift_back=cfg.shift_back,
                merge=False,
                score_mode=cfg.score_mode,
                recv_window=cfg.recv_window,
                query_sketch_mode=cfg.query_sketch_mode,
                receiver_layer_agg=cfg.receiver_layer_agg,
                query_condition_mode=cfg.query_condition_mode,
                receiver_layer_fraction=cfg.receiver_layer_fraction,
                receiver_layer_score=cfg.receiver_layer_score,
            ).to(cfg.device)
            if cfg.profile_cost:
                results = communication_evaluator.test_cost_profile(model_A, cv, limit=cfg.profile_limit, warmup=cfg.profile_warmup)
            else:
                results = communication_evaluator.test(model_A, cv, limit=cfg.limit)
        else:
            if cfg.top_layers > 0:
                cv = CVCommunicator(model_A, model_B, cfg.layer_from, cfg.layer_to, layers_list=cfg.layers_list, top_layers=cfg.top_layers, apply_attn_tracer=True, shift_back=False).to(cfg.device)
                if cfg.random_selection:
                    cfg.layers_list = random.sample(list(range(0, cv.A_num_layers)), int(cfg.top_layers * cv.A_num_layers))
                    logging.info(f"Randomly selected layers list: {cfg.layers_list}")
                else:
                    communication_evaluator.test(model_A, cv, limit=cfg.calib_size, no_wandb=True, do_calc_layer_importance=True)
                    cfg = get_top_layers(communication_evaluator.layer_importance_total, cfg)
            elif cfg.do_layer_curve:
                cv = CVCommunicator(model_A, model_B, cfg.layer_from, cfg.layer_to, layers_list=cfg.layers_list, top_layers=cfg.top_layers, apply_attn_tracer=True, shift_back=False).to(cfg.device)
                communication_evaluator.test(model_A, cv, limit=cfg.calib_size, no_wandb=True, do_calc_layer_importance=True)
                layer_ranking = get_layer_ranking(communication_evaluator.layer_importance_total, cfg)
            if not cfg.do_layer_curve:
                cv = CVCommunicator(model_A, model_B, cfg.layer_from, cfg.layer_to, layers_list=cfg.layers_list, top_layers=cfg.top_layers, apply_attn_tracer=False, shift_back=cfg.shift_back).to(cfg.device)
                if cfg.profile_cost:
                    results = communication_evaluator.test_cost_profile(model_A, cv, limit=cfg.profile_limit, warmup=cfg.profile_warmup)
                else:
                    results = communication_evaluator.test(model_A, cv, limit=cfg.limit)
            else:
                results = []
                for i in range(len(layer_ranking)):
                    layers_list = layer_ranking[:i+1]
                    logging.info(f"Evaluating with layers_list: {layers_list}")
                    cv = CVCommunicator(model_A, model_B, cfg.layer_from, cfg.layer_to, layers_list=layers_list, top_layers=cfg.top_layers, apply_attn_tracer=False, shift_back=cfg.shift_back).to(cfg.device)
                    result = communication_evaluator.test(model_A, cv, limit=cfg.limit)
                    results.append(result)
                logging.info(f"Layer curve results: {results}")
                if cfg.use_wandb:
                    wandb.log({f"layer_curve_results": results})
    if cfg.do_test_ac:
        ac = ActivationCommunicator(model_A, model_B, cfg.layer_k, cfg.layer_j, f=cfg.f).to(cfg.device)
        ac_evaluator = ACEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length)
        results = ac_evaluator.test(model_A, ac, limit=cfg.limit)
    if cfg.do_test_nld:
        nld_evaluator = NLDEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length, cfg.nld_max_tokens_model_A_and_B_phase1, cfg.sender_aware, cfg.nld_receiver_aware)
        if cfg.profile_cost:
            results = nld_evaluator.test_cost_profile(model_A, model_B, limit=cfg.profile_limit, warmup=cfg.profile_warmup)
        else:
            results = nld_evaluator.test(model_A, model_B, limit=cfg.limit)
    if cfg.do_test_text_retrieval:
        text_retrieval_evaluator = QueryAwareTextRetrievalEvaluator(
            evaluator,
            tokenizer,
            cfg.use_wandb,
            cfg.max_input_length,
            top_k=cfg.text_retrieval_top_k,
            chunk_tokens=cfg.text_retrieval_chunk_tokens,
            chunk_stride=cfg.text_retrieval_chunk_stride,
            bm25_k1=cfg.text_retrieval_bm25_k1,
            bm25_b=cfg.text_retrieval_bm25_b,
        )
        if not cfg.profile_cost:
            raise ValueError("--do_test_text_retrieval currently requires --profile_cost")
        results = text_retrieval_evaluator.test_cost_profile(
            model_B,
            limit=cfg.profile_limit,
            warmup=cfg.profile_warmup,
        )
    if cfg.do_test_cipher:
        model_A = CipherAgent(model_A, tokenizer)
        model_B = CipherAgent(model_B, tokenizer)
        cipher_evaluator = CipherEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length, cfg.nld_max_tokens_model_A_and_B_phase1, cfg.sender_aware)
        results = cipher_evaluator.test(model_A, model_B, limit=cfg.limit)
    # Finish W&B run
    if cfg.use_wandb:
        wandb.finish()
    return results


def parse_args() -> AlignConfig:
    parser = argparse.ArgumentParser()
    for field, default in AlignConfig().__dict__.items():
        arg_type = type(default)
        if isinstance(default, bool):
            if default:
                parser.add_argument(f"--no_{field}", dest=field, action="store_false")
            else:
                parser.add_argument(f"--{field}", dest=field, action="store_true")
            parser.set_defaults(**{field: default})
        elif isinstance(default, list):
            element_type = type(default[0])
            parser.add_argument(f"--{field}", type=element_type, default=default, nargs="+")
        else:
            parser.add_argument(f"--{field}", type=arg_type, default=default)
    args = parser.parse_args()
    return AlignConfig(**vars(args))



if __name__ == "__main__":
    config = parse_args()
    main(config)