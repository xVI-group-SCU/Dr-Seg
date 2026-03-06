# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.workers.actor.base import BasePPOActor
from verl.workers.actor.config import ActorConfig


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(
        self, micro_batch: Dict[str, torch.Tensor], temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        vision_inputs = {}
        if "pixel_values" in micro_batch:
            vision_inputs["pixel_values"] = torch.cat(micro_batch["pixel_values"], dim=0)
            vision_inputs["image_grid_thw"] = torch.cat(micro_batch["image_grid_thw"], dim=0)

        if self.config.padding_free:
            # TODO (yaowei): preprocess data for padding_free and ulysses
            raise NotImplementedError
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **vision_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = logprobs_from_logits(logits, responses)  # (bsz, response_length)
            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

        return entropy, log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        self.actor_optimizer.step()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["pixel_values", "image_grid_thw"]
        else:
            non_tensor_select_keys = None

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        for micro_batch in tqdm(micro_batches, desc="Compute log probs", disable=(self.rank != 0)):
            micro_batch.to("cuda")
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            _, log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        grad_attrib_enabled = str(os.getenv("VERL_GRAD_ATTRIB", "0")).lower() not in ("", "0", "false", "no")
        grad_attrib_skip_direction = str(os.getenv("VERL_GRAD_ATTRIB_SKIP_DIRECTION", "0")).lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        components_str = os.getenv("VERL_GRAD_ATTRIB_COMPONENTS", "accuracy_x1,accuracy_x2,accuracy_x3").strip()
        if components_str:
            grad_attrib_components = [c.strip() for c in components_str.split(",") if c.strip()]
        else:
            grad_attrib_components = ["accuracy_x1", "accuracy_x2", "accuracy_x3"]
        serve_component_adv_keys = [
            f"advantages_{c}" for c in grad_attrib_components if f"advantages_{c}" in data.batch.keys()
        ]

        # Raw (pre-serve) reward attribution scenario, if the trainer provides these keys.
        raw_components_str = os.getenv(
            "VERL_GRAD_ATTRIB_RAW_COMPONENTS", "accuracy_raw_x1,accuracy_raw_x2,accuracy_raw_x3"
        ).strip()
        if raw_components_str:
            raw_components = [c.strip() for c in raw_components_str.split(",") if c.strip()]
        else:
            raw_components = []
        raw_component_adv_keys = [f"advantages_{c}" for c in raw_components if f"advantages_{c}" in data.batch.keys()]
        raw_total_adv_key = "advantages_raw_total"
        raw_total_present = raw_total_adv_key in data.batch.keys()

        if grad_attrib_enabled and serve_component_adv_keys:
            select_keys.extend(serve_component_adv_keys)
        if grad_attrib_enabled and raw_total_present and raw_component_adv_keys:
            select_keys.append(raw_total_adv_key)
            select_keys.extend(raw_component_adv_keys)

        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["pixel_values", "image_grid_thw"]
        else:
            non_tensor_select_keys = None

        # TODO (yaowei): support ppo epochs
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        # Resolve attribution controls once per "big batch" (one update_policy call).
        step_val = data.meta_info.get("global_steps", None)
        try:
            step_int = int(step_val) if step_val is not None else -1
        except Exception:
            step_int = -1

        attrib_once = str(os.getenv("VERL_GRAD_ATTRIB_ONCE", "1")).lower() in ("", "1", "true", "yes")
        attrib_steps_filter = os.getenv("VERL_GRAD_ATTRIB_STEPS", "").strip()
        bigbatch_enabled = str(os.getenv("VERL_GRAD_ATTRIB_BIG_BATCH", "0")).lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        bigbatch_only = str(os.getenv("VERL_GRAD_ATTRIB_ONLY_BIG_BATCH", "0")).lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        stream_enabled = str(os.getenv("VERL_GRAD_ATTRIB_STREAM", "1")).lower() not in (
            "",
            "0",
            "false",
            "no",
        )

        attrib_step_enabled = bool(
            grad_attrib_enabled
            and (
                bool(serve_component_adv_keys)
                or (bool(raw_total_present) and bool(raw_component_adv_keys))
            )
        )
        if attrib_step_enabled and attrib_steps_filter:
            try:
                wanted = {int(s) for s in attrib_steps_filter.split(",") if s.strip()}
                if step_int not in wanted:
                    attrib_step_enabled = False
            except Exception:
                pass

        if attrib_step_enabled and attrib_once:
            if not hasattr(self, "_grad_attrib_seen_steps"):
                self._grad_attrib_seen_steps = set()
            if step_int in self._grad_attrib_seen_steps:
                attrib_step_enabled = False

        # Optional minibatch index filter (within this big batch). When empty, run on all mini-batches.
        minibatch_idx_str = os.getenv("VERL_GRAD_ATTRIB_MINIBATCH_IDX", "").strip()
        target_minibatch_idx = None
        if minibatch_idx_str:
            if minibatch_idx_str.lower() in ("all", "*", "any"):
                target_minibatch_idx = None
            else:
                try:
                    target_minibatch_idx = int(minibatch_idx_str)
                except Exception:
                    target_minibatch_idx = None
        else:
            # Default: only compute attribution for the first mini-batch.
            target_minibatch_idx = 0

        did_any_attrib = False
        bigbatch_cache = None
        bigbatch_meta = None

        metrics = defaultdict(list)
        n = len(mini_batches)

        # Optional: compute attribution over the whole big batch (all mini-batches) before updates.
        do_bigbatch_attrib = bool(attrib_step_enabled and bigbatch_enabled)
        if do_bigbatch_attrib:
            try:
                out_path_stream = os.getenv("VERL_GRAD_ATTRIB_OUT", "").strip()
                def _append_jsonl_record(rec: Dict[str, Any]) -> None:
                    if not out_path_stream or self.rank != 0:
                        return
                    try:
                        with open(out_path_stream, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

                def _base_bigbatch_record() -> Dict[str, Any]:
                    return {
                        "timestamp_s": float(time.time()),
                        "global_step": int(step_int),
                        "minibatch_idx": -1,
                        "bigbatch": True,
                        "bigbatch_scope": "all_minibatches",
                        "rank": int(self.rank),
                        "world_size": int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1,
                        "temperature": float(temperature),
                        "gradient_accumulation": int(
                            self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                        ),
                        "global_batch_size_per_device": int(self.config.global_batch_size_per_device),
                        "micro_batch_size_per_device_for_update": int(self.config.micro_batch_size_per_device_for_update),
                        "clip_ratio": float(self.config.clip_ratio),
                        "entropy_coeff": float(self.config.entropy_coeff),
                        "max_grad_norm": float(self.config.max_grad_norm),
                        "use_kl_loss": bool(self.config.use_kl_loss),
                        "kl_loss_coef": float(self.config.kl_loss_coef) if self.config.use_kl_loss else 0.0,
                        "kl_loss_type": str(self.config.kl_loss_type) if self.config.use_kl_loss else "",
                        "reward_breakdown": data.meta_info.get("reward_breakdown", None),
                    }

                cpu_rng_state_big = torch.get_rng_state()
                cuda_rng_state_big = torch.cuda.get_rng_state()

                def _reset_big_rng():
                    torch.set_rng_state(cpu_rng_state_big)
                    torch.cuda.set_rng_state(cuda_rng_state_big)

                def _grad_norm_sq() -> float:
                    device = None
                    total = None
                    for p in self.actor_module.parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        if g.is_sparse:
                            g = g.coalesce().values()
                        if device is None:
                            device = g.device
                            total = torch.zeros((), device=device, dtype=torch.float64)
                        total = total + g.float().pow(2).sum().to(dtype=torch.float64)
                    if total is None:
                        total = torch.zeros((), device="cuda", dtype=torch.float64)
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
                    return float(total.item())

                # Pre-split all mini-batches into micro-batches to compute total count.
                all_micro_batches = []
                total_micro_batches = 0
                for mb in mini_batches:
                    mbs = mb.split(self.config.micro_batch_size_per_device_for_update)
                    all_micro_batches.append(mbs)
                    total_micro_batches += len(mbs)
                total_micro_batches = max(1, total_micro_batches)

                def _backward_full_loss_bigbatch(adv_key: str, adv_key2: Optional[str] = None) -> float:
                    self.actor_optimizer.zero_grad()
                    _reset_big_rng()
                    for mbs in all_micro_batches:
                        for micro_batch in mbs:
                            micro_batch.to("cuda")
                            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                            responses = model_inputs["responses"]
                            response_length = responses.size(1)
                            attention_mask = model_inputs["attention_mask"]
                            response_mask = attention_mask[:, -response_length:]
                            old_log_prob = model_inputs["old_log_probs"]

                            entropy, log_prob = self._forward_micro_batch(model_inputs, temperature=temperature)

                            def _policy_loss_for_adv(k: str) -> torch.Tensor:
                                advantages = model_inputs[k]
                                pg_loss, _, _ = core_algos.compute_policy_loss(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    eos_mask=response_mask,
                                    cliprange=self.config.clip_ratio,
                                )
                                entropy_loss = verl_F.masked_mean(entropy, response_mask)
                                pol = pg_loss - entropy_loss * self.config.entropy_coeff

                                if self.config.use_kl_loss:
                                    ref_log_prob = model_inputs["ref_log_prob"]
                                    kld = core_algos.kl_penalty(
                                        logprob=log_prob,
                                        ref_logprob=ref_log_prob,
                                        kl_penalty=self.config.kl_loss_type,
                                    )
                                    kl_loss = masked_mean(kld, response_mask)
                                    pol = pol + kl_loss * self.config.kl_loss_coef

                                return pol

                            loss = _policy_loss_for_adv(adv_key)
                            if adv_key2 is not None:
                                loss = loss + _policy_loss_for_adv(adv_key2)

                            (loss / float(total_micro_batches)).backward()

                    norm_sq = _grad_norm_sq()
                    self.actor_optimizer.zero_grad()
                    return norm_sq

                bigbatch_cache = {}
                bigbatch_meta = {"total_micro_batches": int(total_micro_batches), "total_minibatches": int(n)}

                attrib_scenarios = []
                if serve_component_adv_keys:
                    attrib_scenarios.append(
                        {
                            "tag": "serve",
                            "total_adv_key": "advantages",
                            "component_adv_keys": list(serve_component_adv_keys),
                        }
                    )
                if raw_total_present and raw_component_adv_keys:
                    attrib_scenarios.append(
                        {
                            "tag": "raw",
                            "total_adv_key": raw_total_adv_key,
                            "component_adv_keys": list(raw_component_adv_keys),
                        }
                    )

                for sc in attrib_scenarios:
                    total_adv_key = sc["total_adv_key"]
                    total_norm_sq = _backward_full_loss_bigbatch(total_adv_key)
                    bigbatch_cache[sc["tag"]] = {
                        "total_adv_key": total_adv_key,
                        "total_norm_sq_debug": total_norm_sq,
                        "components": {},
                    }
                    if stream_enabled:
                        try:
                            rec = _base_bigbatch_record()
                            rec.update(
                                {
                                    "bigbatch_partial": True,
                                    "bigbatch_phase": "total",
                                    "bigbatch_scenario": sc["tag"],
                                    "bigbatch_total_norm_debug_sq": float(total_norm_sq),
                                    "bigbatch_total_norm_debug": float(total_norm_sq) ** 0.5,
                                }
                            )
                            _append_jsonl_record(rec)
                        except Exception:
                            pass
                    for adv_key in sc["component_adv_keys"]:
                        comp_norm_sq = _backward_full_loss_bigbatch(adv_key)
                        bigbatch_cache[sc["tag"]]["components"][adv_key] = {"norm_sq": comp_norm_sq}
                        if stream_enabled:
                            try:
                                suffix = adv_key.replace("advantages_", "")
                                short = suffix.replace("accuracy_", "")
                                rec = _base_bigbatch_record()
                                rec.update(
                                    {
                                        "bigbatch_partial": True,
                                        "bigbatch_phase": "component",
                                        "bigbatch_scenario": sc["tag"],
                                        "bigbatch_component": short,
                                        f"bigbatch_{short}_norm_sq": float(comp_norm_sq),
                                        f"bigbatch_{short}_norm": float(comp_norm_sq) ** 0.5,
                                    }
                                )
                                _append_jsonl_record(rec)
                            except Exception:
                                pass
                        if not grad_attrib_skip_direction:
                            sum_norm_sq = _backward_full_loss_bigbatch(total_adv_key, adv_key2=adv_key)
                            bigbatch_cache[sc["tag"]]["components"][adv_key]["sum_with_total_norm_sq"] = sum_norm_sq
                            if stream_enabled:
                                try:
                                    suffix = adv_key.replace("advantages_", "")
                                    short = suffix.replace("accuracy_", "")
                                    eps = 1e-12
                                    total_norm = float(total_norm_sq) ** 0.5
                                    comp_norm = float(comp_norm_sq) ** 0.5
                                    dot_debug = 0.5 * (float(sum_norm_sq) - float(total_norm_sq) - float(comp_norm_sq))
                                    denom_debug = max(eps, total_norm * max(eps, comp_norm))
                                    cos_raw_debug = dot_debug / denom_debug
                                    cos_debug = max(-1.0, min(1.0, float(cos_raw_debug)))
                                    rec = _base_bigbatch_record()
                                    rec.update(
                                        {
                                            "bigbatch_partial": True,
                                            "bigbatch_phase": "direction",
                                            "bigbatch_scenario": sc["tag"],
                                            "bigbatch_component": short,
                                            f"bigbatch_{short}_sum_with_total_norm_sq": float(sum_norm_sq),
                                            f"bigbatch_{short}_dot_total_debug": float(dot_debug),
                                            f"bigbatch_{short}_cos_total_debug_raw": float(cos_raw_debug),
                                            f"bigbatch_{short}_cos_total_debug": float(cos_debug),
                                            f"bigbatch_{short}_proj_x_total_debug": float(comp_norm * cos_debug),
                                            f"bigbatch_{short}_proj_y_total_debug": float(
                                                comp_norm * (max(0.0, 1.0 - cos_debug * cos_debug) ** 0.5)
                                            ),
                                        }
                                    )
                                    _append_jsonl_record(rec)
                                except Exception:
                                    pass

                # Restore RNG state for the actual training update.
                torch.set_rng_state(cpu_rng_state_big)
                torch.cuda.set_rng_state(cuda_rng_state_big)
            except Exception:
                bigbatch_cache = None
                bigbatch_meta = None
            if bigbatch_only and bigbatch_cache is not None:
                did_any_attrib = True

        for i, mini_batch in enumerate(mini_batches):
            gradient_accumulation = (
                self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
            )
            micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

            # --- Optional: GRPO gradient attribution for reward components (x1/x2/x3) ---
            do_grad_attrib = bool(attrib_step_enabled and not bigbatch_only)
            if do_grad_attrib and target_minibatch_idx is not None and i != target_minibatch_idx:
                do_grad_attrib = False

            attrib_cache = {}
            total_norm_sq_debug = None  # serve total
            if do_grad_attrib:
                did_any_attrib = True
                # Save RNG state to avoid perturbing training randomness, but make attribution runs deterministic.
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state()
                def _reset_debug_rng():
                    # Reset to the same RNG state before each attribution pass so dropout (if any)
                    # matches the real training step, while still not consuming RNG in the real step.
                    torch.set_rng_state(cpu_rng_state)
                    torch.cuda.set_rng_state(cuda_rng_state)

                def _grad_norm_sq() -> float:
                    device = None
                    total = None
                    for p in self.actor_module.parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        if g.is_sparse:
                            g = g.coalesce().values()
                        if device is None:
                            device = g.device
                            total = torch.zeros((), device=device, dtype=torch.float64)
                        total = total + g.float().pow(2).sum().to(dtype=torch.float64)
                    if total is None:
                        total = torch.zeros((), device="cuda", dtype=torch.float64)
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
                    return float(total.item())

                def _backward_full_loss(adv_key: str, adv_key2: Optional[str] = None) -> float:
                    self.actor_optimizer.zero_grad()
                    _reset_debug_rng()
                    for micro_batch in micro_batches:
                        micro_batch.to("cuda")
                        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                        responses = model_inputs["responses"]
                        response_length = responses.size(1)
                        attention_mask = model_inputs["attention_mask"]
                        response_mask = attention_mask[:, -response_length:]
                        old_log_prob = model_inputs["old_log_probs"]

                        # all return: (bsz, response_length)
                        entropy, log_prob = self._forward_micro_batch(model_inputs, temperature=temperature)

                        def _policy_loss_for_adv(k: str) -> torch.Tensor:
                            advantages = model_inputs[k]
                            pg_loss, _, _ = core_algos.compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                eos_mask=response_mask,
                                cliprange=self.config.clip_ratio,
                            )
                            entropy_loss = verl_F.masked_mean(entropy, response_mask)
                            pol = pg_loss - entropy_loss * self.config.entropy_coeff

                            if self.config.use_kl_loss:
                                ref_log_prob = model_inputs["ref_log_prob"]
                                kld = core_algos.kl_penalty(
                                    logprob=log_prob,
                                    ref_logprob=ref_log_prob,
                                    kl_penalty=self.config.kl_loss_type,
                                )
                                kl_loss = masked_mean(kld, response_mask)
                                pol = pol + kl_loss * self.config.kl_loss_coef

                            return pol

                        loss = _policy_loss_for_adv(adv_key)
                        if adv_key2 is not None:
                            loss = loss + _policy_loss_for_adv(adv_key2)

                        (loss / gradient_accumulation).backward()

                    norm_sq = _grad_norm_sq()
                    self.actor_optimizer.zero_grad()
                    return norm_sq

                # Precompute component gradients (and optionally sums with total) before the real optimizer step.
                attrib_scenarios = []
                if serve_component_adv_keys:
                    attrib_scenarios.append(
                        {
                            "tag": "serve",
                            "total_adv_key": "advantages",
                            "component_adv_keys": list(serve_component_adv_keys),
                        }
                    )
                if raw_total_present and raw_component_adv_keys:
                    attrib_scenarios.append(
                        {
                            "tag": "raw",
                            "total_adv_key": raw_total_adv_key,
                            "component_adv_keys": list(raw_component_adv_keys),
                        }
                    )

                for sc in attrib_scenarios:
                    total_adv_key = sc["total_adv_key"]
                    total_norm_sq = _backward_full_loss(total_adv_key)
                    if sc["tag"] == "serve":
                        total_norm_sq_debug = total_norm_sq
                    attrib_cache[sc["tag"]] = {
                        "total_adv_key": total_adv_key,
                        "total_norm_sq_debug": total_norm_sq,
                        "components": {},
                    }
                    for adv_key in sc["component_adv_keys"]:
                        attrib_cache[sc["tag"]]["components"][adv_key] = {"norm_sq": _backward_full_loss(adv_key)}
                        if not grad_attrib_skip_direction:
                            attrib_cache[sc["tag"]]["components"][adv_key]["sum_with_total_norm_sq"] = _backward_full_loss(
                                total_adv_key, adv_key2=adv_key
                            )

                # Restore RNG state for the actual training update.
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state(cuda_rng_state)

            self.actor_optimizer.zero_grad()
            # For jsonl debugging: collect masked advantage stats over this mini-batch.
            adv_stat_keys = ["advantages"] + list(serve_component_adv_keys)
            if raw_total_present and raw_component_adv_keys:
                adv_stat_keys.append(raw_total_adv_key)
                adv_stat_keys.extend(list(raw_component_adv_keys))
            adv_stats = None
            dbg_loss_stats = None
            if do_grad_attrib:
                adv_stats = {
                    k: {
                        "sum": torch.zeros((), device="cuda", dtype=torch.float64),
                        "sumsq": torch.zeros((), device="cuda", dtype=torch.float64),
                        "count": torch.zeros((), device="cuda", dtype=torch.float64),
                        "min": torch.tensor(float("inf"), device="cuda", dtype=torch.float64),
                        "max": torch.tensor(float("-inf"), device="cuda", dtype=torch.float64),
                    }
                    for k in adv_stat_keys
                }
                dbg_loss_stats = {
                    "count": 0,
                    "pg_loss_sum": 0.0,
                    "entropy_loss_sum": 0.0,
                    "pg_clipfrac_sum": 0.0,
                    "ppo_kl_sum": 0.0,
                    "kl_loss_sum": 0.0,
                }
            for micro_batch in tqdm(micro_batches, desc=f"Update policy [{i + 1}/{n}]", disable=(self.rank != 0)):
                micro_batch.to("cuda")
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                responses = model_inputs["responses"]
                response_length = responses.size(1)
                attention_mask = model_inputs["attention_mask"]
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = model_inputs["old_log_probs"]
                advantages = model_inputs["advantages"]

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(model_inputs, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    eos_mask=response_mask,
                    cliprange=clip_ratio,
                )
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    # compute kl loss
                    kld = core_algos.kl_penalty(
                        logprob=log_prob,
                        ref_logprob=ref_log_prob,
                        kl_penalty=self.config.kl_loss_type,
                    )
                    kl_loss = masked_mean(kld, response_mask)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()
                    metrics["actor/kl_coef"] = self.config.kl_loss_coef

                if adv_stats is not None:
                    try:
                        mask = response_mask.bool()
                        if mask.any():
                            for k in adv_stat_keys:
                                v = model_inputs[k]
                                vv = v.masked_select(mask).to(dtype=torch.float64)
                                adv_stats[k]["sum"] += vv.sum()
                                adv_stats[k]["sumsq"] += vv.pow(2).sum()
                                adv_stats[k]["count"] += torch.tensor(float(vv.numel()), device="cuda", dtype=torch.float64)
                                adv_stats[k]["min"] = torch.minimum(adv_stats[k]["min"], vv.min())
                                adv_stats[k]["max"] = torch.maximum(adv_stats[k]["max"], vv.max())
                    except Exception:
                        pass

                if dbg_loss_stats is not None:
                    try:
                        dbg_loss_stats["count"] += 1
                        dbg_loss_stats["pg_loss_sum"] += float(pg_loss.detach().item())
                        dbg_loss_stats["entropy_loss_sum"] += float(entropy_loss.detach().item())
                        dbg_loss_stats["pg_clipfrac_sum"] += float(pg_clipfrac.detach().item())
                        dbg_loss_stats["ppo_kl_sum"] += float(ppo_kl.detach().item())
                        if self.config.use_kl_loss:
                            dbg_loss_stats["kl_loss_sum"] += float(kl_loss.detach().item())
                    except Exception:
                        pass

                loss = policy_loss / gradient_accumulation
                loss.backward()

                batch_metrics = {
                    "actor/entropy_loss": entropy_loss.detach().item(),
                    "actor/pg_loss": pg_loss.detach().item(),
                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    "actor/ppo_kl": ppo_kl.detach().item(),
                }
                append_to_dict(metrics, batch_metrics)

            # Compute and log gradient attribution stats *before* clipping/step.
            if do_grad_attrib and attrib_cache:
                try:
                    # total gradient (the one used for this optimizer step)
                    # NOTE: this is computed before clip_grad_norm_ mutates grads.
                    total_norm_sq = 0.0
                    # local sum of squares
                    local = torch.zeros((), device="cuda", dtype=torch.float64)
                    for p in self.actor_module.parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        if g.is_sparse:
                            g = g.coalesce().values()
                        local = local + g.float().pow(2).sum().to(dtype=torch.float64)
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.SUM)
                    total_norm_sq = float(local.item())
                    total_norm = float(total_norm_sq) ** 0.5
                    total_norm_debug = (float(total_norm_sq_debug) ** 0.5) if total_norm_sq_debug is not None else None

                    if adv_stats is not None and torch.distributed.is_initialized():
                        try:
                            for _, s in adv_stats.items():
                                torch.distributed.all_reduce(s["sum"], op=torch.distributed.ReduceOp.SUM)
                                torch.distributed.all_reduce(s["sumsq"], op=torch.distributed.ReduceOp.SUM)
                                torch.distributed.all_reduce(s["count"], op=torch.distributed.ReduceOp.SUM)
                                torch.distributed.all_reduce(s["min"], op=torch.distributed.ReduceOp.MIN)
                                torch.distributed.all_reduce(s["max"], op=torch.distributed.ReduceOp.MAX)
                        except Exception:
                            pass

                    eps = 1e-12
                    if "serve" in attrib_cache and serve_component_adv_keys:
                        append_to_dict(metrics, {"grad_attrib/total_norm": total_norm})
                        for adv_key in serve_component_adv_keys:
                            comp = attrib_cache.get("serve", {}).get("components", {}).get(adv_key, {})
                            comp_norm_sq = float(comp.get("norm_sq", 0.0))
                            comp_norm = comp_norm_sq ** 0.5

                            # Map "advantages_accuracy_x1" -> "x1" for readability.
                            suffix = adv_key.replace("advantages_", "")
                            short = suffix.replace("accuracy_", "")

                            append_to_dict(metrics, {f"grad_attrib/{short}_norm": comp_norm})

                            if not grad_attrib_skip_direction:
                                sum_norm_sq = float(comp.get("sum_with_total_norm_sq", 0.0))
                                dot = 0.5 * (sum_norm_sq - total_norm_sq - comp_norm_sq)
                                cos = dot / (max(eps, total_norm * comp_norm))
                                cos = max(-1.0, min(1.0, float(cos)))
                                # 2D embedding relative to the total-gradient axis (x-axis).
                                x = comp_norm * cos
                                y = comp_norm * (max(0.0, 1.0 - cos * cos) ** 0.5)

                                append_to_dict(
                                    metrics,
                                    {
                                        f"grad_attrib/{short}_cos_total": cos,
                                        f"grad_attrib/{short}_proj_x": x,
                                        f"grad_attrib/{short}_proj_y": y,
                                    },
                                )

                    if "raw" in attrib_cache and raw_total_present and raw_component_adv_keys:
                        raw_total_norm_sq_debug = float(attrib_cache["raw"].get("total_norm_sq_debug", 0.0) or 0.0)
                        raw_total_norm_debug = raw_total_norm_sq_debug ** 0.5
                        append_to_dict(metrics, {"grad_attrib/raw_total_norm_debug": raw_total_norm_debug})
                        for adv_key in raw_component_adv_keys:
                            comp = attrib_cache.get("raw", {}).get("components", {}).get(adv_key, {})
                            comp_norm_sq = float(comp.get("norm_sq", 0.0))
                            comp_norm = comp_norm_sq ** 0.5
                            suffix = adv_key.replace("advantages_", "")
                            short = suffix.replace("accuracy_", "")
                            append_to_dict(metrics, {f"grad_attrib/{short}_norm": comp_norm})
                            if not grad_attrib_skip_direction:
                                sum_norm_sq = float(comp.get("sum_with_total_norm_sq", 0.0))
                                dot = 0.5 * (sum_norm_sq - raw_total_norm_sq_debug - comp_norm_sq)
                                cos = dot / (max(eps, raw_total_norm_debug * comp_norm))
                                cos = max(-1.0, min(1.0, float(cos)))
                                x = comp_norm * cos
                                y = comp_norm * (max(0.0, 1.0 - cos * cos) ** 0.5)
                                append_to_dict(
                                    metrics,
                                    {
                                        f"grad_attrib/{short}_cos_raw_total_debug": cos,
                                        f"grad_attrib/{short}_proj_x_raw_total_debug": x,
                                        f"grad_attrib/{short}_proj_y_raw_total_debug": y,
                                    },
                                )

                    out_path = os.getenv("VERL_GRAD_ATTRIB_OUT", "").strip()
                    if out_path and self.rank == 0:
                        try:
                            record = {
                                "timestamp_s": float(time.time()),
                                "global_step": int(step_int),
                                "minibatch_idx": int(i),
                                "rank": int(self.rank),
                                "world_size": int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1,
                                "temperature": float(temperature),
                                "gradient_accumulation": int(gradient_accumulation),
                                "global_batch_size_per_device": int(self.config.global_batch_size_per_device),
                                "micro_batch_size_per_device_for_update": int(self.config.micro_batch_size_per_device_for_update),
                                "clip_ratio": float(self.config.clip_ratio),
                                "entropy_coeff": float(self.config.entropy_coeff),
                                "max_grad_norm": float(self.config.max_grad_norm),
                                "use_kl_loss": bool(self.config.use_kl_loss),
                                "kl_loss_coef": float(self.config.kl_loss_coef) if self.config.use_kl_loss else 0.0,
                                "kl_loss_type": str(self.config.kl_loss_type) if self.config.use_kl_loss else "",
                                "total_norm_train": float(total_norm),
                                "total_norm_train_sq": float(total_norm_sq),
                                "total_norm_debug": float(total_norm_debug) if total_norm_debug is not None else None,
                                "total_norm_debug_sq": float(total_norm_sq_debug) if total_norm_sq_debug is not None else None,
                                "reward_breakdown": data.meta_info.get("reward_breakdown", None),
                            }

                            # Raw scenario total gradient norm (debug only).
                            if "raw" in attrib_cache and raw_total_present and raw_component_adv_keys:
                                raw_total_norm_sq_debug = float(attrib_cache["raw"].get("total_norm_sq_debug", 0.0) or 0.0)
                                record["raw_total_norm_debug_sq"] = raw_total_norm_sq_debug
                                record["raw_total_norm_debug"] = raw_total_norm_sq_debug ** 0.5

                            if dbg_loss_stats is not None and dbg_loss_stats.get("count", 0) > 0:
                                c = float(dbg_loss_stats["count"])
                                record.update(
                                    {
                                        "loss/pg_loss_mean": dbg_loss_stats["pg_loss_sum"] / c,
                                        "loss/entropy_loss_mean": dbg_loss_stats["entropy_loss_sum"] / c,
                                        "loss/pg_clipfrac_mean": dbg_loss_stats["pg_clipfrac_sum"] / c,
                                        "loss/ppo_kl_mean": dbg_loss_stats["ppo_kl_sum"] / c,
                                        "loss/kl_loss_mean": (dbg_loss_stats["kl_loss_sum"] / c)
                                        if self.config.use_kl_loss
                                        else 0.0,
                                    }
                                )

                            if adv_stats is not None:
                                for k, s in adv_stats.items():
                                    cnt = float(s["count"].item())
                                    if cnt <= 0:
                                        mean = std = vmin = vmax = 0.0
                                    else:
                                        sumv = float(s["sum"].item())
                                        sumsq = float(s["sumsq"].item())
                                        mean = sumv / cnt
                                        var = max(0.0, sumsq / cnt - mean * mean)
                                        std = var ** 0.5
                                        vmin = float(s["min"].item())
                                        vmax = float(s["max"].item())
                                    short_k = k.replace("advantages_", "").replace("accuracy_", "")
                                    record.update(
                                        {
                                            f"adv/{short_k}_count": cnt,
                                            f"adv/{short_k}_mean": mean,
                                            f"adv/{short_k}_std": std,
                                            f"adv/{short_k}_min": vmin,
                                            f"adv/{short_k}_max": vmax,
                                        }
                                    )

                            # Serve scenario: component attribution relative to the actual training total gradient.
                            if "serve" in attrib_cache and serve_component_adv_keys:
                                for adv_key in serve_component_adv_keys:
                                    suffix = adv_key.replace("advantages_", "")
                                    short = suffix.replace("accuracy_", "")
                                    comp = attrib_cache.get("serve", {}).get("components", {}).get(adv_key, {})
                                    comp_norm_sq = float(comp.get("norm_sq", 0.0))
                                    comp_norm = comp_norm_sq ** 0.5
                                    record[f"{short}_norm"] = comp_norm
                                    record[f"{short}_norm_sq"] = comp_norm_sq
                                    if not grad_attrib_skip_direction:
                                        sum_norm_sq = float(comp.get("sum_with_total_norm_sq", 0.0))
                                        record[f"{short}_sum_with_total_norm_sq"] = sum_norm_sq
                                        # Prefer a mathematically-consistent dot/cos computed fully from debug passes.
                                        if total_norm_sq_debug is not None:
                                            dot_debug = 0.5 * (sum_norm_sq - float(total_norm_sq_debug) - comp_norm_sq)
                                            denom_debug = max(
                                                eps, (float(total_norm_sq_debug) ** 0.5) * max(eps, comp_norm)
                                            )
                                            cos_raw_debug = dot_debug / denom_debug
                                            cos_debug = max(-1.0, min(1.0, float(cos_raw_debug)))
                                            record[f"{short}_dot_total_debug"] = float(dot_debug)
                                            record[f"{short}_cos_total_debug_raw"] = float(cos_raw_debug)
                                            record[f"{short}_cos_total_debug"] = float(cos_debug)
                                            record[f"{short}_proj_x_total_debug"] = float(comp_norm * cos_debug)
                                            record[f"{short}_proj_y_total_debug"] = float(
                                                comp_norm * (max(0.0, 1.0 - cos_debug * cos_debug) ** 0.5)
                                            )
                                        # Also record the mixed (train+debug) estimate for diagnosing inconsistency.
                                        dot_mixed = 0.5 * (sum_norm_sq - total_norm_sq - comp_norm_sq)
                                        denom_mixed = max(eps, total_norm * max(eps, comp_norm))
                                        cos_raw_mixed = dot_mixed / denom_mixed
                                        record[f"{short}_dot_total_mixed"] = float(dot_mixed)
                                        record[f"{short}_cos_total_mixed_raw"] = float(cos_raw_mixed)
                                        record[f"{short}_cos_total_mixed"] = float(
                                            max(-1.0, min(1.0, float(cos_raw_mixed)))
                                        )

                            # Raw scenario: component attribution relative to a "raw total reward" GRPO gradient (debug only).
                            if "raw" in attrib_cache and raw_total_present and raw_component_adv_keys:
                                raw_total_norm_sq_debug = float(attrib_cache["raw"].get("total_norm_sq_debug", 0.0) or 0.0)
                                raw_total_norm_debug = raw_total_norm_sq_debug ** 0.5
                                for adv_key in raw_component_adv_keys:
                                    suffix = adv_key.replace("advantages_", "")
                                    short = suffix.replace("accuracy_", "")
                                    comp = attrib_cache.get("raw", {}).get("components", {}).get(adv_key, {})
                                    comp_norm_sq = float(comp.get("norm_sq", 0.0))
                                    comp_norm = comp_norm_sq ** 0.5
                                    record[f"{short}_norm"] = comp_norm
                                    record[f"{short}_norm_sq"] = comp_norm_sq
                                    if not grad_attrib_skip_direction:
                                        sum_norm_sq = float(comp.get("sum_with_total_norm_sq", 0.0))
                                        record[f"{short}_sum_with_total_norm_sq"] = sum_norm_sq
                                        dot_debug = 0.5 * (sum_norm_sq - raw_total_norm_sq_debug - comp_norm_sq)
                                        denom_debug = max(eps, raw_total_norm_debug * max(eps, comp_norm))
                                        cos_raw_debug = dot_debug / denom_debug
                                        cos_debug = max(-1.0, min(1.0, float(cos_raw_debug)))
                                        record[f"{short}_dot_total_debug"] = float(dot_debug)
                                        record[f"{short}_cos_total_debug_raw"] = float(cos_raw_debug)
                                        record[f"{short}_cos_total_debug"] = float(cos_debug)
                                        record[f"{short}_proj_x_total_debug"] = float(comp_norm * cos_debug)
                                        record[f"{short}_proj_y_total_debug"] = float(
                                            comp_norm * (max(0.0, 1.0 - cos_debug * cos_debug) ** 0.5)
                                        )
                            with open(out_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        except Exception:
                            pass

                    # Optional breakpoint (safe only for single-process training).
                    pdb_flag = str(os.getenv("VERL_GRAD_ATTRIB_PDB", "0")).strip().lower() not in (
                        "",
                        "0",
                        "false",
                        "no",
                    )
                    if pdb_flag and self.rank == 0:
                        try:
                            if (not torch.distributed.is_initialized()) or torch.distributed.get_world_size() == 1:
                                import pdb

                                pdb.set_trace()
                        except Exception:
                            pass
                except Exception:
                    pass

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        # Mark step as "seen" only after the whole big batch finishes, so we can attribute all mini-batches.
        if attrib_step_enabled and attrib_once and did_any_attrib and step_int >= 0:
            try:
                self._grad_attrib_seen_steps.add(step_int)
            except Exception:
                pass

        # Optional: dump big-batch attribution record (all mini-batches combined).
        if (
            do_bigbatch_attrib
            and bigbatch_cache is not None
            and (self.rank == 0)
            and str(os.getenv("VERL_GRAD_ATTRIB_OUT", "").strip()) != ""
        ):
            try:
                out_path = os.getenv("VERL_GRAD_ATTRIB_OUT", "").strip()
                record = {
                    "timestamp_s": float(time.time()),
                    "global_step": int(step_int),
                    "minibatch_idx": -1,
                    "bigbatch": True,
                    "bigbatch_scope": "all_minibatches",
                    "rank": int(self.rank),
                    "world_size": int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1,
                    "temperature": float(temperature),
                    "gradient_accumulation": int(
                        self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                    ),
                    "global_batch_size_per_device": int(self.config.global_batch_size_per_device),
                    "micro_batch_size_per_device_for_update": int(self.config.micro_batch_size_per_device_for_update),
                    "clip_ratio": float(self.config.clip_ratio),
                    "entropy_coeff": float(self.config.entropy_coeff),
                    "max_grad_norm": float(self.config.max_grad_norm),
                    "use_kl_loss": bool(self.config.use_kl_loss),
                    "kl_loss_coef": float(self.config.kl_loss_coef) if self.config.use_kl_loss else 0.0,
                    "kl_loss_type": str(self.config.kl_loss_type) if self.config.use_kl_loss else "",
                    "reward_breakdown": data.meta_info.get("reward_breakdown", None),
                }
                if isinstance(bigbatch_meta, dict):
                    record["bigbatch_micro_batches"] = int(bigbatch_meta.get("total_micro_batches", 0))
                    record["bigbatch_minibatches"] = int(bigbatch_meta.get("total_minibatches", 0))

                eps = 1e-12
                if "serve" in bigbatch_cache and serve_component_adv_keys:
                    total_norm_sq = float(bigbatch_cache["serve"].get("total_norm_sq_debug", 0.0))
                    total_norm = total_norm_sq ** 0.5
                    record["bigbatch_total_norm_debug_sq"] = total_norm_sq
                    record["bigbatch_total_norm_debug"] = total_norm
                    for adv_key in serve_component_adv_keys:
                        suffix = adv_key.replace("advantages_", "")
                        short = suffix.replace("accuracy_", "")
                        comp = bigbatch_cache.get("serve", {}).get("components", {}).get(adv_key, {})
                        comp_norm_sq = float(comp.get("norm_sq", 0.0))
                        comp_norm = comp_norm_sq ** 0.5
                        record[f"bigbatch_{short}_norm"] = comp_norm
                        record[f"bigbatch_{short}_norm_sq"] = comp_norm_sq
                        if not grad_attrib_skip_direction:
                            sum_norm_sq = float(comp.get("sum_with_total_norm_sq", 0.0))
                            record[f"bigbatch_{short}_sum_with_total_norm_sq"] = sum_norm_sq
                            dot_debug = 0.5 * (sum_norm_sq - total_norm_sq - comp_norm_sq)
                            denom_debug = max(eps, total_norm * max(eps, comp_norm))
                            cos_raw_debug = dot_debug / denom_debug
                            cos_debug = max(-1.0, min(1.0, float(cos_raw_debug)))
                            record[f"bigbatch_{short}_dot_total_debug"] = float(dot_debug)
                            record[f"bigbatch_{short}_cos_total_debug_raw"] = float(cos_raw_debug)
                            record[f"bigbatch_{short}_cos_total_debug"] = float(cos_debug)
                            record[f"bigbatch_{short}_proj_x_total_debug"] = float(comp_norm * cos_debug)
                            record[f"bigbatch_{short}_proj_y_total_debug"] = float(
                                comp_norm * (max(0.0, 1.0 - cos_debug * cos_debug) ** 0.5)
                            )

                if "raw" in bigbatch_cache and raw_total_present and raw_component_adv_keys:
                    total_norm_sq = float(bigbatch_cache["raw"].get("total_norm_sq_debug", 0.0))
                    total_norm = total_norm_sq ** 0.5
                    record["bigbatch_raw_total_norm_debug_sq"] = total_norm_sq
                    record["bigbatch_raw_total_norm_debug"] = total_norm
                    for adv_key in raw_component_adv_keys:
                        suffix = adv_key.replace("advantages_", "")
                        short = suffix.replace("accuracy_", "")
                        comp = bigbatch_cache.get("raw", {}).get("components", {}).get(adv_key, {})
                        comp_norm_sq = float(comp.get("norm_sq", 0.0))
                        comp_norm = comp_norm_sq ** 0.5
                        record[f"bigbatch_{short}_norm"] = comp_norm
                        record[f"bigbatch_{short}_norm_sq"] = comp_norm_sq
                        if not grad_attrib_skip_direction:
                            sum_norm_sq = float(comp.get("sum_with_total_norm_sq", 0.0))
                            record[f"bigbatch_{short}_sum_with_total_norm_sq"] = sum_norm_sq
                            dot_debug = 0.5 * (sum_norm_sq - total_norm_sq - comp_norm_sq)
                            denom_debug = max(eps, total_norm * max(eps, comp_norm))
                            cos_raw_debug = dot_debug / denom_debug
                            cos_debug = max(-1.0, min(1.0, float(cos_raw_debug)))
                            record[f"bigbatch_{short}_dot_total_debug"] = float(dot_debug)
                            record[f"bigbatch_{short}_cos_total_debug_raw"] = float(cos_raw_debug)
                            record[f"bigbatch_{short}_cos_total_debug"] = float(cos_debug)
                            record[f"bigbatch_{short}_proj_x_total_debug"] = float(comp_norm * cos_debug)
                            record[f"bigbatch_{short}_proj_y_total_debug"] = float(
                                comp_norm * (max(0.0, 1.0 - cos_debug * cos_debug) ** 0.5)
                            )

                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass

        # Optional breakpoint at the end of a big batch (safe only for single-process training).
        pdb_big_batch_flag = str(os.getenv("VERL_GRAD_ATTRIB_PDB_BIG_BATCH_END", "0")).strip().lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        if pdb_big_batch_flag and self.rank == 0 and did_any_attrib:
            try:
                if (not torch.distributed.is_initialized()) or torch.distributed.get_world_size() == 1:
                    import pdb

                    pdb.set_trace()
            except Exception:
                pass

        self.actor_optimizer.zero_grad()
        return metrics
