import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import wandb
from layer_importance import calc_layer_importance
from collections import defaultdict
import time
import os
import json
import copy
from pathlib import Path

RECEIVER_AWARE_SCORE_MODES = {
    "receiver",
    "receiver_oracle",
    "receiver_x_value_norm",
    "receiver_value_norm",
    "receiver_recency",
    "receiver_recency_prior",
}

PROTOCOL_ACCOUNTING = {
    "index_dtype": "uint32",
    "query_sketch_header": "2xuint32",
    "query_sketch_layer_descriptor": "4xuint32",
    "kv_header": "2xuint32",
    "kv_layer_descriptor": "5xuint32",
}


def _compute_receiver_token_importance(cv, input_ids_B, out_A_past_key_values):
    if getattr(cv, "score_mode", "") == "receiver_oracle":
        cv.compute_oracle_receiver_importance(input_ids_B, out_A_past_key_values)
    else:
        cv.compute_receiver_importance(input_ids_B, out_A_past_key_values)


def _current_run_dir():
    """Directory of the active log.log (set up in com.py via setup_logging).

    Used to drop per-sample result files next to each run's log without
    threading an extra path argument through every evaluator constructor.
    """
    for h in logging.getLogger().handlers:
        base = getattr(h, "baseFilename", None)
        if base:
            return os.path.dirname(base)
    return None


def _cuda_available_for_model(model):
    try:
        return next(model.parameters()).is_cuda
    except StopIteration:
        return torch.cuda.is_available()


def _sync_if_cuda(model):
    if _cuda_available_for_model(model):
        torch.cuda.synchronize()


def _peak_memory_gb(model):
    if not _cuda_available_for_model(model):
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _reset_peak_memory(model):
    if _cuda_available_for_model(model):
        torch.cuda.reset_peak_memory_stats()

QA_INSTRUCTION = "Directly answer the question based on the context passage, no explanation is needed."
MATH_INSTRUCTION = "Answer the math problem step by step."
CODE_INSTRUCTION = "Complete ONLY THE NEXT LINE of the code snippet based on the context."
SUMMARIZE_INSTRUCTION = "Summarize the following content concisely with one sentence."

SKTLINE_QA_MSG_TEMPLATE = "Instruction: {instruction} Context: {context} Question: {question}"
SKTLINE_MATH_MSG_TEMPLATE = "Instruction: {instruction} Hint: {hint} Question: {question}"
SKTLINE_CODE_MSG_TEMPLATE = "Instruction: {instruction} Context: {context} Code Snippet: {code_snippet}"
SKTLINE_SUMMARIZE_MSG_TEMPLATE = "Instruction: {instruction} Content part 1: {content_part_1} Content part 2: {content_part_2}"

BASELINE_QA_MSG_TEMPLATE = "Instruction: {instruction} Question: {question}"
BASELINE_MATH_MSG_TEMPLATE = "Instruction: {instruction} Question: {question}"
BASELINE_CODE_MSG_TEMPLATE = "Instruction: {instruction} Code Snippet: {code_snippet}"
BASELINE_SUMMARIZE_MSG_TEMPLATE = "Instruction: {instruction} Content: {content_part_2}"

COMMUNICATION_QA_MSG_TEMPLATE_A = "Instruction: {instruction} Context: {context}"
COMMUNICATION_QA_MSG_TEMPLATE_B = "Instruction: {instruction} Question: {question}"
COMMUNICATION_MATH_MSG_TEMPLATE_A = "Instruction: {instruction} Hint: {hint}"
COMMUNICATION_MATH_MSG_TEMPLATE_B = "Instruction: {instruction} Question: {question}"
COMMUNICATION_CODE_MSG_TEMPLATE_A = "Instruction: {instruction} Context: {context}"
COMMUNICATION_CODE_MSG_TEMPLATE_B = "Instruction: {instruction} Code Snippet: {code_snippet}"
COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_A = "Instruction: {instruction} Content part 1: {content_part_1}"
COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_B = "Instruction: {instruction} Content part 2: {content_part_2}"

SENDER_QA_INSTRUCTION = "Summarize the context passage in a concise way, as it will be used by another agent to answer the question."
SENDER_MATH_INSTRUCTION = "Summarize the hint in a concise way, as it will be used by another agent to answer the question."
SENDER_CODE_INSTRUCTION = "Summarize the code snippet in a concise way, as it will be used by another agent to complete the code."
SENDER_SUMMARIZE_INSTRUCTION = "Summarize the content in a concise way, as it will be used by another agent to understand the content."

THINK_MODEL_LIST = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]

def is_think_model(model):
    model_name = str(getattr(model, "name", ""))
    normalized_names = {
        model_name.lower(),
        Path(model_name).name.lower(),
    }
    for think_model in THINK_MODEL_LIST:
        think_model = think_model.lower()
        if think_model in normalized_names or Path(think_model).name in normalized_names:
            return True
    return False

def apply_chat_template(evaluator, tokenizer, msg, model, context=False):
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    if is_think_model(model):
        think_model_prefix = "</think>\n\n"
        if not context and evaluator.name not in ["tipsheets", "repobench"]:
            # for tipsheets, we do not add "The answer is: " as there is already
            # an answer prefix in the data
            if evaluator.name == "countries":
                think_model_prefix += "The only country is:"
            else:
                think_model_prefix += "The answer is: "
        
        if context:
            think_token_id = tokenizer.convert_tokens_to_ids("<think>")
            # remove the think token from the input ids
            input_ids = input_ids[input_ids != think_token_id].unsqueeze(0)
        else:
            end_think_token_id = tokenizer.encode(think_model_prefix, add_special_tokens=False)
            input_ids = torch.cat([input_ids, torch.tensor([end_think_token_id], device=model.device)], dim=-1)
    return input_ids

class SkylineEvaluator:
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length):
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.use_wandb = use_wandb
        self.max_input_length = max_input_length
        self.name = "skyline"
        self.generate_args = {
            "max_new_tokens": self.evaluator.max_tokens,
            "temperature": 1.0,
            "num_beams": 1,
            "top_p": None,
            "top_k": None,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id
        }
    
    def truncate_input(self, input_ids):
        if input_ids.shape[-1] > self.max_input_length and self.evaluator.truncate_input:
            half = int(self.max_input_length / 2)
            input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=-1)
        return input_ids

    def prepare_input_ids(self, item, model):
        if hasattr(self.evaluator, "tmath"):
            msg = SKTLINE_MATH_MSG_TEMPLATE.format(instruction=MATH_INSTRUCTION, hint=item["prompt_A"], question=item["prompt_B"])
        elif hasattr(self.evaluator, "repobench"):
            msg = SKTLINE_CODE_MSG_TEMPLATE.format(instruction=CODE_INSTRUCTION, context=item["prompt_A"], code_snippet=item["prompt_B"])
        elif hasattr(self.evaluator, "sasum"):
            msg = SKTLINE_SUMMARIZE_MSG_TEMPLATE.format(instruction=SUMMARIZE_INSTRUCTION, content_part_1=item["prompt_A"], content_part_2=item["prompt_B"])
        else:
            msg = SKTLINE_QA_MSG_TEMPLATE.format(instruction=QA_INSTRUCTION, context=item["prompt_A"], question=item["prompt_B"])
        input_ids = apply_chat_template(self.evaluator, self.tokenizer, msg, model)
        
        # truncate in the middle of the input
        input_ids = self.truncate_input(input_ids)
        return input_ids

    def get_response(self, output, context_length, truncate_response=True):
        if truncate_response:
            response = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(output, skip_special_tokens=True)
        return response

    def inference(self, model, item):
        input_ids = self.prepare_input_ids(item, model)

        output = model.generate(
            input_ids, 
            attention_mask=torch.ones_like(input_ids),
            **self.generate_args
        )[0]
        
        context_length = input_ids.shape[-1]
        response = self.get_response(output, context_length)
        return response

    def _test(self, model, limit=None):
        progress_bar = tqdm(self.evaluator, desc=f"{self.name} result: 0.0000")
            
        for i, item in enumerate(progress_bar):
            if limit is not None and i >= limit:
                break
            response = self.inference(model, item)
            
            self.evaluator.evaluate_item(item, response)
            
            result = self.evaluator.get_result()
            progress_bar.set_description(f"{self.name} result: {result:.4f}")
            
        result = self.evaluator.get_result()
        return result
    
    @torch.no_grad()
    def test(self, model_A, model_B, limit=None):
        tic = time.time()
        result_A = self._test(model_A, limit)
        toc = time.time()
        time_A = toc - tic
        tic = time.time()
        result_B = self._test(model_B, limit)
        toc = time.time()
        time_B = toc - tic
        
        if self.use_wandb:
            wandb.log({f"{self.name}_result_A": result_A, f"{self.name}_result_B": result_B, f"{self.name}_time_A": time_A, f"{self.name}_time_B": time_B})
        logging.info(f"{self.name} result A: {result_A:.4f}, {self.name} result B: {result_B:.4f}, {self.name} time A: {time_A:.2f}s, {self.name} time B: {time_B:.2f}s")
        return result_A, result_B

class BaselineEvaluator(SkylineEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)
        self.name = "baseline"
        
    def prepare_input_ids(self, item, model):
        if hasattr(self.evaluator, "tmath"):
            msg = BASELINE_MATH_MSG_TEMPLATE.format(instruction=MATH_INSTRUCTION, question=item["prompt_B"])
        elif hasattr(self.evaluator, "repobench"):
            msg = BASELINE_CODE_MSG_TEMPLATE.format(instruction=CODE_INSTRUCTION, code_snippet=item["prompt_B"])
        elif hasattr(self.evaluator, "sasum"):
            msg = BASELINE_SUMMARIZE_MSG_TEMPLATE.format(instruction=SUMMARIZE_INSTRUCTION, content_part_2=item["prompt_B"])
        else:
            msg = BASELINE_QA_MSG_TEMPLATE.format(instruction=QA_INSTRUCTION, question=item["prompt_B"])
        input_ids = apply_chat_template(self.evaluator, self.tokenizer, msg, model)
        
        # truncate in the middle of the input
        input_ids = self.truncate_input(input_ids)
        return input_ids

class CommunicationEvaluator(SkylineEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)
        self.name = "communication"
        self.layer_importance_total = defaultdict(list)
    
    def truncate_input(self, input_ids_A, input_ids_B):
        if input_ids_A.shape[-1] + input_ids_B.shape[-1] > self.max_input_length and self.evaluator.truncate_input:
            half = int((self.max_input_length - input_ids_B.shape[-1]) / 2)
            input_ids_A = torch.cat([input_ids_A[:, :half], input_ids_A[:, -half:]], dim=-1)
        return input_ids_A, input_ids_B

    def prepare_input_ids(self, item, model_A, model_B):
        if hasattr(self.evaluator, "tmath"):
            msg_A = COMMUNICATION_MATH_MSG_TEMPLATE_A.format(instruction=MATH_INSTRUCTION, hint=item["prompt_A"])
        elif hasattr(self.evaluator, "repobench"):
            msg_A = COMMUNICATION_CODE_MSG_TEMPLATE_A.format(instruction=CODE_INSTRUCTION, context=item["prompt_A"])
        elif hasattr(self.evaluator, "sasum"):
            msg_A = COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_A.format(instruction=SUMMARIZE_INSTRUCTION, content_part_1=item["prompt_A"])
        else:
            msg_A = COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=QA_INSTRUCTION, context=item["prompt_A"])
        input_ids_A = apply_chat_template(self.evaluator, self.tokenizer, msg_A, model_A, context=True)

        if hasattr(self.evaluator, "tmath"):
            msg_B = COMMUNICATION_MATH_MSG_TEMPLATE_B.format(instruction=MATH_INSTRUCTION, question=item["prompt_B"])
        elif hasattr(self.evaluator, "repobench"):
            msg_B = COMMUNICATION_CODE_MSG_TEMPLATE_B.format(instruction=CODE_INSTRUCTION, code_snippet=item["prompt_B"])
        elif hasattr(self.evaluator, "sasum"):
            msg_B = COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_B.format(instruction=SUMMARIZE_INSTRUCTION, content_part_2=item["prompt_B"])
        else:
            msg_B = COMMUNICATION_QA_MSG_TEMPLATE_B.format(instruction=QA_INSTRUCTION, question=item["prompt_B"])
        input_ids_B = apply_chat_template(self.evaluator, self.tokenizer, msg_B, model_B)
        
        # truncate in the middle of the input
        input_ids_A, input_ids_B = self.truncate_input(input_ids_A, input_ids_B)

        return input_ids_A, input_ids_B

    def inference(self, model, cv, item):
        input_ids_A, input_ids_B = self.prepare_input_ids(item, cv.A, cv.B)

        out_A = model(
            input_ids=input_ids_A, 
            use_cache=True, 
            return_dict=True
        )
        out_A_past_key_values = out_A.past_key_values

        # Receiver-aware scoring (Pass 1). The default path sends a query-only
        # Q sketch from B to A; receiver_oracle retains the old full-A-KV upper bound.
        if getattr(cv, "score_mode", "value_norm") in RECEIVER_AWARE_SCORE_MODES:
            _compute_receiver_token_importance(cv, input_ids_B, out_A_past_key_values)

        output = cv.generate(
            input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            out_A_past_key_values=out_A_past_key_values,
            **self.generate_args
        )[0]
        
        context_length = input_ids_B.shape[-1]
        response = self.get_response(output, context_length)
        return response

    def inference_with_cost(self, model, cv, item):
        """Run one communication sample and return response plus timing/payload stats.

        This path is only used by --profile_cost. It mirrors inference() but
        synchronizes CUDA around coarse phases so the cost table can separate the
        receiver-aware scoring overhead from regular generation.
        """
        _reset_peak_memory(model)
        _sync_if_cuda(model)
        t0 = time.perf_counter()
        input_ids_A, input_ids_B = self.prepare_input_ids(item, cv.A, cv.B)
        _sync_if_cuda(model)
        t_prepare = time.perf_counter() - t0

        t0 = time.perf_counter()
        out_A = model(
            input_ids=input_ids_A,
            use_cache=True,
            return_dict=True,
        )
        _sync_if_cuda(model)
        t_a_prefill = time.perf_counter() - t0
        out_A_past_key_values = out_A.past_key_values

        t_receiver_score = 0.0
        if getattr(cv, "score_mode", "value_norm") in RECEIVER_AWARE_SCORE_MODES:
            t0 = time.perf_counter()
            _compute_receiver_token_importance(cv, input_ids_B, out_A_past_key_values)
            _sync_if_cuda(model)
            t_receiver_score = time.perf_counter() - t0

        t0 = time.perf_counter()
        output = cv.generate(
            input_ids_B,
            attention_mask=torch.ones_like(input_ids_B),
            out_A_past_key_values=out_A_past_key_values,
            **self.generate_args
        )[0]
        _sync_if_cuda(model)
        t_generate_total = time.perf_counter() - t0

        context_length = input_ids_B.shape[-1]
        response = self.get_response(output, context_length)
        output_tokens = int(max(output.shape[-1] - context_length, 0))
        kv_cost = getattr(cv, "last_kv_cost", None) or {}
        protocol_timing = getattr(cv, "last_protocol_timing", None) or {}
        t_sender_compress = float(protocol_timing.get("t_sender_compress", 0.0))
        row = {
            "protocol_version": getattr(cv, "protocol_version", None),
            "ctx_tokens_A": int(input_ids_A.shape[-1]),
            "query_tokens_B": int(input_ids_B.shape[-1]),
            "output_tokens": output_tokens,
            "t_prepare_inputs": round(float(t_prepare), 6),
            "t_a_prefill": round(float(t_a_prefill), 6),
            "t_receiver_score": round(float(t_receiver_score), 6),
            "t_b_query_prefill": round(float(protocol_timing.get("t_b_query_prefill", 0.0)), 6),
            "t_sender_score": round(float(protocol_timing.get("t_sender_score", 0.0)), 6),
            "t_budget_compute": round(float(protocol_timing.get("t_budget_compute", 0.0)), 6),
            "t_oracle_kv_copy": round(float(protocol_timing.get("t_oracle_kv_copy", 0.0)), 6),
            "t_sender_compress": round(t_sender_compress, 6),
            "t_generate_total": round(float(t_generate_total), 6),
            "t_b_generate": round(float(max(t_generate_total - t_sender_compress, 0.0)), 6),
            "t_total": round(float(t_prepare + t_a_prefill + t_receiver_score + t_generate_total), 6),
            "peak_mem_gb": round(float(_peak_memory_gb(model)), 6) if _peak_memory_gb(model) is not None else None,
            "budget": round(float(getattr(cv, "last_kept_ratio")), 6) if getattr(cv, "last_kept_ratio", None) is not None else None,
            "query_budget": round(float(getattr(cv, "last_query_budget")), 6) if getattr(cv, "last_query_budget", None) is not None else None,
        }
        for key, value in kv_cost.items():
            if isinstance(value, float):
                row[key] = round(value, 6)
            else:
                row[key] = value
        return response, row

    def _test(self, model_A, cv, limit=None, do_calc_layer_importance=False):
        progress_bar = tqdm(self.evaluator, desc=f"{self.name} result: 0.0000", disable=do_calc_layer_importance)

        # collect per-sample scores for the real eval pass (skip calibration pass)
        per_sample = None if do_calc_layer_importance else []

        for i, item in enumerate(progress_bar):
            if limit is not None and i >= limit:
                break
            response = self.inference(model_A, cv, item)

            if do_calc_layer_importance:
                cv.calc_attn_weights_from_qk()
                self.layer_importance_total = calc_layer_importance(cv.B_attn_weights, model_A.name, self.layer_importance_total)
            
            prev_total = self.evaluator.f1_total
            self.evaluator.evaluate_item(item, response)

            if per_sample is not None:
                # evaluate_item does f1_total += score, f1_count += 1, so the
                # delta is exactly this sample's score (F1 / EM / ROUGE-L recall).
                score = self.evaluator.f1_total - prev_total
                sid = None
                try:
                    sid = item.get("_id", item.get("id", None))
                except AttributeError:
                    sid = None
                row = {"idx": i, "id": sid, "score": round(float(score), 6)}
                # achieved transmitted-KV fraction and per-query budget (budget-aware runs)
                kept = getattr(cv, "last_kept_ratio", None)
                if kept is not None:
                    row["budget"] = round(float(kept), 6)
                qb = getattr(cv, "last_query_budget", None)
                if qb is not None:
                    row["query_budget"] = round(float(qb), 6)
                coverage_target = getattr(cv, "last_coverage_target", None)
                if coverage_target is not None:
                    row["coverage_target"] = round(float(coverage_target), 6)
                coverage_achieved = getattr(cv, "last_coverage_achieved", None)
                if coverage_achieved is not None:
                    row["coverage_achieved"] = round(float(coverage_achieved), 6)
                coverage_satisfied = getattr(cv, "last_coverage_satisfied_ratio", None)
                if coverage_satisfied is not None:
                    row["coverage_satisfied_layer_ratio"] = round(float(coverage_satisfied), 6)
                kv_cost = getattr(cv, "last_kv_cost", None) or {}
                for key in (
                    "protocol_version",
                    "kv_tokens_sent",
                    "kv_bytes_sent",
                    "query_sketch_bytes",
                    "query_sketch_metadata_bytes",
                    "selection_index_bytes",
                    "kv_metadata_bytes",
                    "a_to_b_communication_bytes",
                    "b_to_a_communication_bytes",
                    "communication_metadata_bytes",
                    "total_communication_bytes",
                ):
                    if key in kv_cost:
                        row[key] = kv_cost[key]
                per_sample.append(row)

            result = self.evaluator.get_result()
            progress_bar.set_description(f"{self.name} result: {result:.4f}")
            
        result = self.evaluator.get_result()
        if per_sample is not None:
            self._dump_per_sample(per_sample, cv)
        return result

    def _dump_per_sample(self, per_sample, cv):
        run_dir = _current_run_dir()
        if run_dir is None:
            return
        meta = {
            "protocol_version": getattr(cv, "protocol_version", None),
            "protocol_accounting": PROTOCOL_ACCOUNTING,
            "dataset": getattr(self.evaluator, "name", None),
            "score_mode": getattr(cv, "score_mode", None),
            "recv_window": getattr(cv, "recv_window", None),
            "merge": getattr(cv, "merge", None),
            "merge_mode": getattr(cv, "merge_mode", None),
            "merge_ratio": getattr(cv, "merge_ratio", None),
            "budget_mode": getattr(cv, "budget_mode", None),
            "budget_min": getattr(cv, "budget_min", None),
            "budget_max": getattr(cv, "budget_max", None),
            "coverage_tau": getattr(cv, "coverage_tau", None),
            "coverage_scale": getattr(cv, "coverage_scale", None),
            "coverage_tau_mode": getattr(cv, "coverage_tau_mode", None),
            "coverage_tau_min": getattr(cv, "coverage_tau_min", None),
            "coverage_tau_max": getattr(cv, "coverage_tau_max", None),
            "n": len(per_sample),
        }
        path = os.path.join(run_dir, "per_sample.jsonl")
        try:
            with open(path, "w") as f:
                f.write(json.dumps({"_meta": meta}) + "\n")
                for row in per_sample:
                    f.write(json.dumps(row) + "\n")
            logging.info(f"per-sample scores written to {path}")
        except OSError as e:
            logging.warning(f"failed to write per-sample scores: {e}")
    
    @torch.no_grad()
    def test(self, model_A, cv, limit=None, do_calc_layer_importance=False, no_wandb=False):
        tic = time.time()
        result = self._test(model_A, cv, limit, do_calc_layer_importance)
        toc = time.time()
        time_used = toc - tic
        if self.use_wandb and not no_wandb and not do_calc_layer_importance:
            wandb.log({f"{self.name}_result": result, f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result

    @torch.no_grad()
    def test_cost_profile(self, model_A, cv, limit=50, warmup=5):
        measured = []
        warmup = int(max(warmup or 0, 0))
        limit = int(max(limit or 0, 0))
        total_needed = None if limit <= 0 else warmup + limit
        pbar = tqdm(self.evaluator, desc=f"{self.name} cost-profile")
        for i, item in enumerate(pbar):
            if total_needed is not None and i >= total_needed:
                break
            response, row = self.inference_with_cost(model_A, cv, item)
            if i < warmup:
                pbar.set_description(f"{self.name} cost-profile warmup {i + 1}/{warmup}")
                continue

            prev_total = self.evaluator.f1_total
            self.evaluator.evaluate_item(item, response)
            score = self.evaluator.f1_total - prev_total
            sid = item.get("_id", item.get("id", None)) if hasattr(item, "get") else None
            row.update({
                "idx": i - warmup,
                "source_idx": i,
                "id": sid,
                "score": round(float(score), 6),
            })
            measured.append(row)
            target = "all" if limit <= 0 else str(limit)
            pbar.set_description(f"{self.name} cost-profile [{len(measured)}/{target}]")

        self._dump_cost_profile(measured, cv, warmup)
        result = self.evaluator.get_result()
        logging.info(f"{self.name} cost profile result: {result:.4f}, measured={len(measured)}, warmup={warmup}")
        return result

    def _dump_cost_profile(self, rows, cv, warmup):
        run_dir = _current_run_dir()
        if run_dir is None:
            return
        meta = {
            "protocol_version": getattr(cv, "protocol_version", None),
            "protocol_accounting": PROTOCOL_ACCOUNTING,
            "dataset": getattr(self.evaluator, "name", None),
            "score_mode": getattr(cv, "score_mode", None),
            "recv_window": getattr(cv, "recv_window", None),
            "merge": getattr(cv, "merge", None),
            "merge_mode": getattr(cv, "merge_mode", None),
            "merge_ratio": getattr(cv, "merge_ratio", None),
            "budget_mode": getattr(cv, "budget_mode", None),
            "budget_min": getattr(cv, "budget_min", None),
            "budget_max": getattr(cv, "budget_max", None),
            "coverage_tau": getattr(cv, "coverage_tau", None),
            "coverage_scale": getattr(cv, "coverage_scale", None),
            "coverage_tau_mode": getattr(cv, "coverage_tau_mode", None),
            "coverage_tau_min": getattr(cv, "coverage_tau_min", None),
            "coverage_tau_max": getattr(cv, "coverage_tau_max", None),
            "warmup": warmup,
            "n": len(rows),
        }
        profile_path = os.path.join(run_dir, "cost_profile.jsonl")
        summary_path = os.path.join(run_dir, "cost_summary.json")
        try:
            with open(profile_path, "w") as f:
                f.write(json.dumps({"_meta": meta}) + "\n")
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            summary = self._summarize_cost_profile(rows, meta)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
                f.write("\n")
            logging.info(f"cost profile written to {profile_path}")
            logging.info(f"cost summary written to {summary_path}")
        except OSError as e:
            logging.warning(f"failed to write cost profile: {e}")

    def _summarize_cost_profile(self, rows, meta):
        def mean(key):
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 6) if vals else None

        keys = [
            "score",
            "budget",
            "query_budget",
            "kv_token_ratio",
            "kv_byte_ratio",
            "kv_tokens_sent",
            "kv_bytes_sent",
            "query_sketch_tokens",
            "query_sketch_layers",
            "query_sketch_elements",
            "query_sketch_bytes",
            "query_sketch_metadata_bytes",
            "oracle_full_kv_bytes",
            "selection_index_bytes",
            "kv_metadata_bytes",
            "a_to_b_communication_bytes",
            "b_to_a_communication_bytes",
            "communication_metadata_bytes",
            "total_communication_bytes",
            "coverage_target",
            "coverage_achieved",
            "coverage_satisfied_layer_ratio",
            "ctx_tokens_A",
            "query_tokens_B",
            "output_tokens",
            "t_prepare_inputs",
            "t_a_prefill",
            "t_receiver_score",
            "t_b_query_prefill",
            "t_sender_score",
            "t_budget_compute",
            "t_oracle_kv_copy",
            "t_sender_compress",
            "t_generate_total",
            "t_b_generate",
            "t_total",
            "peak_mem_gb",
        ]
        summary = {"_meta": meta}
        for key in keys:
            summary[f"{key}_mean"] = mean(key)
        return summary

    # ---------- Step 2b: online progressive communication ----------

    @torch.no_grad()
    def _generate_uncertainty(self, cv, input_ids_B, out_A_past_key_values):
        """Generate B's answer at the current budget and return (response, uncertainty, budget).

        Uncertainty signals (computed from the step logits of the greedy decode):
          - ent_first / ent_mean : predictive entropy of the first / mean of all generated tokens
          - margin_first/margin_mean: top-1 minus top-2 probability (low = unsure)
        These are exactly the signals a real receiver could compute at inference time
        to decide whether to request more KV.
        """
        gen_args = dict(self.generate_args)
        gen_args.update(return_dict_in_generate=True, output_scores=True)
        out = cv.generate(
            input_ids_B,
            attention_mask=torch.ones_like(input_ids_B),
            out_A_past_key_values=out_A_past_key_values,
            **gen_args,
        )
        context_length = input_ids_B.shape[-1]
        response = self.get_response(out.sequences[0], context_length)
        budget = getattr(cv, "last_kept_ratio", None)

        ents, margins = [], []
        for s in out.scores:
            logp = torch.log_softmax(s[0].float(), dim=-1)
            p = logp.exp()
            ents.append(float(-(p * logp).sum()))
            top2 = torch.topk(p, 2).values
            margins.append(float(top2[0] - top2[1]))
        if not ents:  # no token generated -> maximally unsure
            unc = {"ent_first": 20.0, "ent_mean": 20.0, "margin_first": 0.0, "margin_mean": 0.0}
        else:
            unc = {
                "ent_first": ents[0],
                "ent_mean": sum(ents) / len(ents),
                "margin_first": margins[0],
                "margin_mean": sum(margins) / len(margins),
            }
        return response, unc, budget

    @torch.no_grad()
    def _test_progressive(self, model_A, cv, ladder, limit=None):
        """For each sample, generate at every budget rung and record
        {score, budget, uncertainty} so any stop-threshold theta can be swept
        offline (scripts/analyze_progressive_online.py). The receiver-aware
        importance is query-dependent but budget-independent, so it is computed
        once per sample and reused across rungs (only merge_ratio changes)."""
        assert getattr(cv, "score_mode", None) == "receiver", "progressive needs score_mode=receiver"
        ladder = sorted(float(r) for r in ladder)
        per_sample = []
        pbar = tqdm(self.evaluator, desc=f"{self.name} progressive")
        for i, item in enumerate(pbar):
            if limit is not None and i >= limit:
                break
            input_ids_A, input_ids_B = self.prepare_input_ids(item, cv.A, cv.B)
            out_A = model_A(input_ids=input_ids_A, use_cache=True, return_dict=True)
            base_pkv = out_A.past_key_values
            _compute_receiver_token_importance(cv, input_ids_B, base_pkv)  # once, r-independent

            rungs = []
            for r in ladder:
                cv.merge_ratio = float(r)
                # KV-sufficiency signal: B's attention over the COMPRESSED A-context
                suff = cv.compute_context_attention(
                    input_ids_B, cv.prepare_key_cache(copy.deepcopy(base_pkv)))
                pkv = copy.deepcopy(base_pkv)  # generate appends to its own copy
                resp, unc, budget = self._generate_uncertainty(cv, input_ids_B, pkv)
                prev_total, prev_count = self.evaluator.f1_total, self.evaluator.f1_count
                self.evaluator.evaluate_item(item, resp)
                score = self.evaluator.f1_total - prev_total
                self.evaluator.f1_total, self.evaluator.f1_count = prev_total, prev_count  # don't pollute
                rungs.append({
                    "r": float(r),
                    "budget": round(float(budget), 6) if budget is not None else None,
                    "score": round(float(score), 6),
                    **{k: round(float(v), 6) for k, v in unc.items()},
                    **{k: round(float(v), 6) for k, v in suff.items()},
                })
            sid = item.get("_id", item.get("id", None)) if hasattr(item, "get") else None
            per_sample.append({"idx": i, "id": sid, "rungs": rungs})
            pbar.set_description(f"{self.name} progressive [{i+1}]")

        self._dump_progressive(per_sample, cv, ladder)
        return per_sample

    def _dump_progressive(self, per_sample, cv, ladder):
        run_dir = _current_run_dir()
        if run_dir is None:
            return
        meta = {
            "dataset": getattr(self.evaluator, "name", None),
            "score_mode": getattr(cv, "score_mode", None),
            "recv_window": getattr(cv, "recv_window", None),
            "ladder": list(ladder),
            "signals": ["ent_first", "ent_mean", "margin_first", "margin_mean", "ctx_mass", "ctx_conc"],
            "n": len(per_sample),
        }
        path = os.path.join(run_dir, "per_sample_prog.jsonl")
        try:
            with open(path, "w") as f:
                f.write(json.dumps({"_meta": meta}) + "\n")
                for row in per_sample:
                    f.write(json.dumps(row) + "\n")
            logging.info(f"progressive per-sample records written to {path}")
        except OSError as e:
            logging.warning(f"failed to write progressive records: {e}")

    @torch.no_grad()
    def test_progressive(self, model_A, cv, ladder, limit=None):
        tic = time.time()
        self._test_progressive(model_A, cv, ladder, limit)
        logging.info(f"progressive done in {time.time()-tic:.1f}s, ladder={sorted(float(r) for r in ladder)}")

    # ---------- 牌2: single-shot budget-prediction feature dump ----------

    @torch.no_grad()
    def dump_pass1_features(self, model_A, cv, limit=None):
        """For each sample, run only A-prefill + receiver scoring (Pass-1, no
        generation) and dump the dimensionless budget-prediction features.
        These join (by idx) with the probe per_sample.jsonl (oracle min budget)
        to train a single-shot budget predictor offline."""
        assert getattr(cv, "score_mode", None) == "receiver", "feature dump needs score_mode=receiver"
        per_sample = []
        pbar = tqdm(self.evaluator, desc=f"{self.name} pass1-features")
        for i, item in enumerate(pbar):
            if limit is not None and i >= limit:
                break
            input_ids_A, input_ids_B = self.prepare_input_ids(item, cv.A, cv.B)
            out_A = model_A(input_ids=input_ids_A, use_cache=True, return_dict=True)
            _compute_receiver_token_importance(cv, input_ids_B, out_A.past_key_values)
            feat = cv.compute_pass1_features()
            sid = item.get("_id", item.get("id", None)) if hasattr(item, "get") else None
            per_sample.append({"idx": i, "id": sid, **feat})
        self._dump_features(per_sample, cv)
        return per_sample

    def _dump_features(self, per_sample, cv):
        run_dir = _current_run_dir()
        if run_dir is None:
            return
        meta = {
            "dataset": getattr(self.evaluator, "name", None),
            "score_mode": getattr(cv, "score_mode", None),
            "recv_window": getattr(cv, "recv_window", None),
            "n": len(per_sample),
            "features": [k for k in per_sample[0].keys() if k not in ("idx", "id")] if per_sample else [],
        }
        path = os.path.join(run_dir, "per_sample_feat.jsonl")
        try:
            with open(path, "w") as f:
                f.write(json.dumps({"_meta": meta}) + "\n")
                for row in per_sample:
                    f.write(json.dumps(row) + "\n")
            logging.info(f"pass1 features written to {path}")
        except OSError as e:
            logging.warning(f"failed to write pass1 features: {e}")

class ACEvaluator(CommunicationEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)
        self.name = "ac"

    def inference(self, model, ac, item):
        input_ids_A, input_ids_B = self.prepare_input_ids(item, ac.A, ac.B)

        out_A = model(
            input_ids=input_ids_A, 
            use_cache=True, 
            output_hidden_states=True, 
            return_dict=True
        )

        output = ac.generate(
            input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            h_A=out_A.hidden_states,
            **self.generate_args
        )[0]
        
        context_length = input_ids_B.shape[-1]
        response = self.get_response(output, context_length)
        return response

    def _test(self, model_A, ac, limit=None):
        progress_bar = tqdm(self.evaluator, desc=f"{self.name} result: 0.0000")

        for i, item in enumerate(progress_bar):
            if limit is not None and i >= limit:
                break
            response = self.inference(model_A, ac, item)

            self.evaluator.evaluate_item(item, response)
            
            result = self.evaluator.get_result()
            progress_bar.set_description(f"{self.name} result: {result:.4f}")
            
        result = self.evaluator.get_result()
        return result
    
    @torch.no_grad()
    def test(self, model_A, ac, limit=None):
        tic = time.time()
        result = self._test(model_A, ac, limit)
        toc = time.time()
        time_used = toc - tic
        if self.use_wandb:
            wandb.log({f"{self.name}_result": result, f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result

REFINE_TMPL = "{prompt}\nYour previous answer:\n{self_answer}\nOther agents' answers (for your consideration):\n{others}\nIf needed, revise your answer. Your new answer is:"


class NLDEvaluator(CommunicationEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length, max_tokens_A_model_phase1, sender_aware=False):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)
        self.name = "nld"
        self.max_tokens_phase_1 = max_tokens_A_model_phase1
        self.sender_aware = sender_aware

    def prepare_input_ids(self, item, model_A, model_B):
        if self.sender_aware:
            if hasattr(self.evaluator, "tmath"):
                msg_A = COMMUNICATION_MATH_MSG_TEMPLATE_A.format(instruction=SENDER_MATH_INSTRUCTION, hint=item["prompt_A"])
            elif hasattr(self.evaluator, "repobench"):
                msg_A = COMMUNICATION_CODE_MSG_TEMPLATE_A.format(instruction=SENDER_CODE_INSTRUCTION, context=item["prompt_A"])
            elif hasattr(self.evaluator, "sasum"):
                msg_A = COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_A.format(instruction=SENDER_SUMMARIZE_INSTRUCTION, content_part_1=item["prompt_A"])
            else:
                msg_A = COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=SENDER_QA_INSTRUCTION, context=item["prompt_A"])
        else:
            if hasattr(self.evaluator, "tmath"):
                msg_A = COMMUNICATION_MATH_MSG_TEMPLATE_A.format(instruction=MATH_INSTRUCTION, hint=item["prompt_A"])
            elif hasattr(self.evaluator, "repobench"):
                msg_A = COMMUNICATION_CODE_MSG_TEMPLATE_A.format(instruction=CODE_INSTRUCTION, context=item["prompt_A"])
            elif hasattr(self.evaluator, "sasum"):
                msg_A = COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_A.format(instruction=SUMMARIZE_INSTRUCTION, content_part_1=item["prompt_A"])
            else:
                msg_A = COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=QA_INSTRUCTION, context=item["prompt_A"])
        input_ids_A = apply_chat_template(self.evaluator, self.tokenizer, msg_A, model_A)

        if hasattr(self.evaluator, "tmath"):
            msg_B = COMMUNICATION_MATH_MSG_TEMPLATE_B.format(instruction=MATH_INSTRUCTION, question=item["prompt_B"])
        elif hasattr(self.evaluator, "repobench"):
            msg_B = COMMUNICATION_CODE_MSG_TEMPLATE_B.format(instruction=CODE_INSTRUCTION, code_snippet=item["prompt_B"])
        elif hasattr(self.evaluator, "sasum"):
            msg_B = COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_B.format(instruction=SUMMARIZE_INSTRUCTION, content_part_2=item["prompt_B"])
        else:
            msg_B = COMMUNICATION_QA_MSG_TEMPLATE_B.format(instruction=QA_INSTRUCTION, question=item["prompt_B"])
        input_ids_B = apply_chat_template(self.evaluator, self.tokenizer, msg_B, model_B)
        
        # truncate in the middle of the input
        input_ids_A, input_ids_B = self.truncate_input(input_ids_A, input_ids_B)

        return input_ids_A, input_ids_B, msg_B

    def truncate_input_nld(self, input_ids):
        if input_ids.shape[-1] > self.max_input_length and self.evaluator.truncate_input:
            half = int(self.max_input_length / 2)
            input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=-1)
        return input_ids

    def prepare_input_ids_nld(self, prompt: str, self_answer: str, others: str, model):
        msg = REFINE_TMPL.format(prompt=prompt, self_answer=self_answer, others=others)
        input_ids = apply_chat_template(self.evaluator, self.tokenizer, msg, model)
        
        # truncate in the middle of the input
        input_ids = self.truncate_input_nld(input_ids)
        return input_ids

    def inference(self, model_A, model_B, item):
        input_ids_A, input_ids_B, msg_B = self.prepare_input_ids(item, model_A, model_B)
        # overwrite max_new_tokens for model A and model B for phase 1
        self.generate_args["max_new_tokens"] = self.max_tokens_phase_1

        output = model_A.generate(
            input_ids_A, 
            attention_mask=torch.ones_like(input_ids_A),
            **self.generate_args,
        )[0]

        context_length = input_ids_A.shape[-1]
        initial_answer_A = self.get_response(output, context_length)

        output = model_B.generate(
            input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            **self.generate_args
        )[0]
        context_length = input_ids_B.shape[-1]
        initial_answer_B = self.get_response(output, context_length)

        # restore generation for new tokens
        self.generate_args["max_new_tokens"] = self.evaluator.max_tokens

        input_ids = self.prepare_input_ids_nld(msg_B, initial_answer_B, initial_answer_A, model_B)
        output = model_B.generate(
            input_ids, 
            attention_mask=torch.ones_like(input_ids),
            **self.generate_args
        )[0]
        context_length = input_ids.shape[-1]
        response = self.get_response(output, context_length)
        return response

    def inference_with_cost(self, model_A, model_B, item):
        """Run one NLD sample and return response plus timing/token stats.

        NLD communicates natural-language text from A to B, so the main payload
        is the number of A answer tokens and bytes inserted into B's refinement
        prompt. We also record total prompt/generation tokens because NLD spends
        compute on three generation calls, unlike one-shot KV communication.
        """
        _reset_peak_memory(model_A)
        _sync_if_cuda(model_A)
        t0 = time.perf_counter()
        input_ids_A, input_ids_B, msg_B = self.prepare_input_ids(item, model_A, model_B)
        _sync_if_cuda(model_A)
        t_prepare_inputs = time.perf_counter() - t0

        self.generate_args["max_new_tokens"] = self.max_tokens_phase_1

        t0 = time.perf_counter()
        output_A = model_A.generate(
            input_ids_A,
            attention_mask=torch.ones_like(input_ids_A),
            **self.generate_args,
        )[0]
        _sync_if_cuda(model_A)
        t_model_a_phase1 = time.perf_counter() - t0
        ctx_A = input_ids_A.shape[-1]
        initial_answer_A = self.get_response(output_A, ctx_A)
        answer_tokens_A = int(max(output_A.shape[-1] - ctx_A, 0))

        t0 = time.perf_counter()
        output_B_initial = model_B.generate(
            input_ids_B,
            attention_mask=torch.ones_like(input_ids_B),
            **self.generate_args,
        )[0]
        _sync_if_cuda(model_B)
        t_model_b_phase1 = time.perf_counter() - t0
        ctx_B = input_ids_B.shape[-1]
        initial_answer_B = self.get_response(output_B_initial, ctx_B)
        answer_tokens_B_initial = int(max(output_B_initial.shape[-1] - ctx_B, 0))

        self.generate_args["max_new_tokens"] = self.evaluator.max_tokens

        t0 = time.perf_counter()
        refine_input_ids = self.prepare_input_ids_nld(msg_B, initial_answer_B, initial_answer_A, model_B)
        _sync_if_cuda(model_B)
        t_prepare_refine = time.perf_counter() - t0

        t0 = time.perf_counter()
        output = model_B.generate(
            refine_input_ids,
            attention_mask=torch.ones_like(refine_input_ids),
            **self.generate_args,
        )[0]
        _sync_if_cuda(model_B)
        t_model_b_refine = time.perf_counter() - t0
        refine_ctx = refine_input_ids.shape[-1]
        response = self.get_response(output, refine_ctx)
        output_tokens = int(max(output.shape[-1] - refine_ctx, 0))

        t_total = t_prepare_inputs + t_model_a_phase1 + t_model_b_phase1 + t_prepare_refine + t_model_b_refine
        row = {
            "ctx_tokens_A": int(ctx_A),
            "query_tokens_B": int(ctx_B),
            "nld_answer_tokens_A": answer_tokens_A,
            "nld_answer_tokens_B_initial": answer_tokens_B_initial,
            "nld_refine_input_tokens": int(refine_ctx),
            "nld_text_payload_tokens": answer_tokens_A,
            "nld_text_payload_bytes": len(initial_answer_A.encode("utf-8")),
            "output_tokens": output_tokens,
            "t_prepare_inputs": round(float(t_prepare_inputs), 6),
            "t_model_a_phase1": round(float(t_model_a_phase1), 6),
            "t_model_b_phase1": round(float(t_model_b_phase1), 6),
            "t_prepare_refine": round(float(t_prepare_refine), 6),
            "t_model_b_refine": round(float(t_model_b_refine), 6),
            "t_total": round(float(t_total), 6),
            "peak_mem_gb": round(float(_peak_memory_gb(model_A)), 6) if _peak_memory_gb(model_A) is not None else None,
        }
        return response, row

    def _test(self, model_A, model_B, limit=None):
        progress_bar = tqdm(self.evaluator, desc=f"{self.name} result: 0.0000")

        for i, item in enumerate(progress_bar):
            if limit is not None and i >= limit:
                break
            try:
                response = self.inference(model_A, model_B, item)
            except Exception as e:
                logging.error(f"Error during inference: {e}")
                continue
            
            self.evaluator.evaluate_item(item, response)
            
            result = self.evaluator.get_result()
            progress_bar.set_description(f"{self.name} result: {result:.4f}")
            
        result = self.evaluator.get_result()
        return result

    def _summarize_nld_cost_profile(self, rows, meta):
        def mean(key):
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 6) if vals else None

        keys = [
            "score",
            "ctx_tokens_A",
            "query_tokens_B",
            "nld_answer_tokens_A",
            "nld_answer_tokens_B_initial",
            "nld_refine_input_tokens",
            "nld_text_payload_tokens",
            "nld_text_payload_bytes",
            "output_tokens",
            "t_prepare_inputs",
            "t_model_a_phase1",
            "t_model_b_phase1",
            "t_prepare_refine",
            "t_model_b_refine",
            "t_total",
            "peak_mem_gb",
        ]
        summary = {"_meta": meta}
        for key in keys:
            summary[f"{key}_mean"] = mean(key)
        return summary

    def _dump_nld_cost_profile(self, rows, warmup):
        run_dir = _current_run_dir()
        if run_dir is None:
            return
        meta = {
            "dataset": getattr(self.evaluator, "name", None),
            "method": self.name,
            "sender_aware": self.sender_aware,
            "max_tokens_phase_1": self.max_tokens_phase_1,
            "warmup": warmup,
            "n": len(rows),
        }
        profile_path = os.path.join(run_dir, "cost_profile.jsonl")
        summary_path = os.path.join(run_dir, "cost_summary.json")
        try:
            with open(profile_path, "w") as f:
                f.write(json.dumps({"_meta": meta}) + "\n")
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            summary = self._summarize_nld_cost_profile(rows, meta)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
                f.write("\n")
            logging.info(f"nld cost profile written to {profile_path}")
            logging.info(f"nld cost summary written to {summary_path}")
        except OSError as e:
            logging.warning(f"failed to write nld cost profile: {e}")

    @torch.no_grad()
    def test_cost_profile(self, model_A, model_B, limit=50, warmup=5):
        measured = []
        warmup = int(max(warmup or 0, 0))
        limit = int(max(limit or 0, 0))
        total_needed = None if limit <= 0 else warmup + limit
        pbar = tqdm(self.evaluator, desc=f"{self.name} cost-profile")
        for i, item in enumerate(pbar):
            if total_needed is not None and i >= total_needed:
                break
            try:
                response, row = self.inference_with_cost(model_A, model_B, item)
            except Exception as e:
                logging.error(f"Error during NLD cost inference: {e}")
                continue
            if i < warmup:
                pbar.set_description(f"{self.name} cost-profile warmup {i + 1}/{warmup}")
                continue

            prev_total = self.evaluator.f1_total
            self.evaluator.evaluate_item(item, response)
            score = self.evaluator.f1_total - prev_total
            sid = item.get("_id", item.get("id", None)) if hasattr(item, "get") else None
            row.update({
                "idx": i - warmup,
                "source_idx": i,
                "id": sid,
                "score": round(float(score), 6),
            })
            measured.append(row)
            target = "all" if limit <= 0 else str(limit)
            pbar.set_description(f"{self.name} cost-profile [{len(measured)}/{target}]")

        self._dump_nld_cost_profile(measured, warmup)
        result = self.evaluator.get_result()
        logging.info(f"{self.name} cost profile result: {result:.4f}, measured={len(measured)}, warmup={warmup}")
        return result
    
    @torch.no_grad()
    def test(self, model_A, model_B, limit=None):
        tic = time.time()
        result = self._test(model_A, model_B, limit)
        toc = time.time()
        time_used = toc - tic
        if self.use_wandb:
            wandb.log({f"{self.name}_result": result, f"{self.name}_time": time_used})
        logging.info(f"{self.name} result: {result:.4f}, {self.name} time: {time_used:.2f}s")
        return result



class CipherEvaluator(NLDEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length, max_tokens_A_model_phase1, sender_aware=False):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length, max_tokens_A_model_phase1, sender_aware)
        self.name = "cipher"
        self.max_tokens_phase_1 = max_tokens_A_model_phase1

    def prepare_inputs_embeds_cipher(self, prompt: str, cipher_embeds_self: torch.Tensor, cipher_embeds_others: torch.Tensor, model):
        msg = REFINE_TMPL.format(prompt=prompt, self_answer="<SELF_ANS>", others="<OTHERS_ANS>")
        input_ids = apply_chat_template(self.evaluator, self.tokenizer, msg, model.model)[0]

        sentinel_positions = (input_ids == model.SELF_ID).nonzero(as_tuple=False), (input_ids == model.OTHERS_ID).nonzero(as_tuple=False)
        self_pos = sentinel_positions[0][0].item()
        others_pos = sentinel_positions[1][0].item()
        if not (0 <= self_pos < others_pos < input_ids.numel()):
            raise RuntimeError("Unexpected sentinel positions")

        prefix_ids  = input_ids[:self_pos]
        middle_ids  = input_ids[self_pos+1:others_pos]
        suffix_ids  = input_ids[others_pos+1:]

        prefix_emb = F.embedding(prefix_ids, model.embed_weight)
        middle_emb = F.embedding(middle_ids, model.embed_weight)
        suffix_emb = F.embedding(suffix_ids, model.embed_weight)

        cipher_embeds_self = cipher_embeds_self.squeeze(0)
        cipher_embeds_others = cipher_embeds_others.squeeze(0)
        
        inputs_embeds = torch.cat(
            [prefix_emb, cipher_embeds_self, middle_emb, cipher_embeds_others, suffix_emb],
            dim=0
        ).unsqueeze(0)  
        # truncate in the middle of the input
        assert inputs_embeds.shape[1] <= self.max_input_length, "Input length is too long"
        return inputs_embeds

    def inference(self, model_A, model_B, item):
        input_ids_A, input_ids_B, msg_B = self.prepare_input_ids(item, model_A.model, model_B.model)
        # overwrite max_new_tokens for model A and model B for phase 1
        self.generate_args["max_new_tokens"] = self.max_tokens_phase_1

        cipher_embeds_A = model_A.cipher_generate(
            input_ids=input_ids_A, 
            attention_mask=torch.ones_like(input_ids_A),
            **self.generate_args,
        )

        cipher_embeds_B = model_B.cipher_generate(
            input_ids=input_ids_B, 
            attention_mask=torch.ones_like(input_ids_B),
            **self.generate_args
        )

        # restore generation for new tokens
        self.generate_args["max_new_tokens"] = self.evaluator.max_tokens

        inputs_embeds = self.prepare_inputs_embeds_cipher(msg_B, cipher_embeds_B, cipher_embeds_A, model_B)
        output = model_B.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=torch.ones_like(inputs_embeds[..., 0]),
            **self.generate_args
        )[0]
        response = self.get_response(output, None, truncate_response=False)
        return response
