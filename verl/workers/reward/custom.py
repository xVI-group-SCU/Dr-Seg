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

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import math_compute_score, r1v_compute_score, seg_compute_score, seg_strict_compute_score, vision_reasoner_compute_score, dr_seg_compute_score

import os


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "seg":
            self.compute_score = seg_compute_score
        elif compute_score == "seg_strict":
            self.compute_score = seg_strict_compute_score
        elif compute_score == "vision_reasoner":
            self.compute_score = vision_reasoner_compute_score
        elif compute_score == "dr_seg":
            self.compute_score = dr_seg_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        def _env_flag(name: str, default: str = "0") -> bool:
            v = str(os.getenv(name, default)).strip().lower()
            return v not in ("", "0", "false", "no")

        want_breakdown = _env_flag("VERL_REWARD_BREAKDOWN") or _env_flag("VERL_GRAD_ATTRIB")
        breakdown_tensors = {}
        raw_tensors = {}
        breakdown_sums = {}
        breakdown_sumsq = {}
        breakdown_counts = {}
        extra_sums = {}
        extra_sumsq = {}
        extra_count = 0
        if want_breakdown:
            # Keep shapes identical to reward_tensor: (bsz, response_len)
            breakdown_tensors = {
                "format": torch.zeros_like(reward_tensor),
                "non_repeat": torch.zeros_like(reward_tensor),
                "accuracy": torch.zeros_like(reward_tensor),
                # accuracy reward components from the quantile service (r1/r2/r3)
                "x1": torch.zeros_like(reward_tensor),
                "x2": torch.zeros_like(reward_tensor),
                "x3": torch.zeros_like(reward_tensor),
            }
            # Raw (pre-serve) accuracy components (x1/x2/x3) before quantile service transform.
            raw_tensors = {
                "raw_x1": torch.zeros_like(reward_tensor),
                "raw_x2": torch.zeros_like(reward_tensor),
                "raw_x3": torch.zeros_like(reward_tensor),
            }
            breakdown_sums = {k: 0.0 for k in breakdown_tensors.keys()}
            breakdown_sumsq = {k: 0.0 for k in breakdown_tensors.keys()}
            breakdown_counts = {k: 0 for k in breakdown_tensors.keys()}
            extra_sums = {"x1_raw": 0.0, "x2_raw": 0.0, "x3_raw": 0.0, "query_ok": 0.0}
            extra_sumsq = {"x1_raw": 0.0, "x2_raw": 0.0, "x3_raw": 0.0}

        already_print = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # ground_truth = data_item.non_tensor_batch["answer"]
            ground_truth = data_item.non_tensor_batch["solution"]
            # print(ground_truth,response_str)

            score = self.compute_score(response_str, ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            if want_breakdown:
                bd = getattr(self.compute_score, "latest_breakdown", None)
                if isinstance(bd, dict):
                    try:
                        fmt = float(bd.get("format_reward", 0.0))
                        nrep = float(bd.get("non_repeat_reward", 0.0))
                        acc = float(bd.get("accuracy_reward", 0.0))
                        x1 = float(bd.get("accuracy/r1", 0.0))
                        x2 = float(bd.get("accuracy/r2", 0.0))
                        x3 = float(bd.get("accuracy/r3", 0.0))
                    except Exception:
                        fmt = nrep = acc = x1 = x2 = x3 = 0.0
                    try:
                        x1_raw = float(bd.get("accuracy/x1", 0.0))
                        x2_raw = float(bd.get("accuracy/x2", 0.0))
                        x3_raw = float(bd.get("accuracy/x3", 0.0))
                    except Exception:
                        x1_raw = x2_raw = x3_raw = 0.0

                    breakdown_tensors["format"][i, valid_response_length - 1] = fmt
                    breakdown_tensors["non_repeat"][i, valid_response_length - 1] = nrep
                    breakdown_tensors["accuracy"][i, valid_response_length - 1] = acc
                    breakdown_tensors["x1"][i, valid_response_length - 1] = x1
                    breakdown_tensors["x2"][i, valid_response_length - 1] = x2
                    breakdown_tensors["x3"][i, valid_response_length - 1] = x3
                    raw_tensors["raw_x1"][i, valid_response_length - 1] = x1_raw
                    raw_tensors["raw_x2"][i, valid_response_length - 1] = x2_raw
                    raw_tensors["raw_x3"][i, valid_response_length - 1] = x3_raw

                    breakdown_sums["format"] += fmt
                    breakdown_sums["non_repeat"] += nrep
                    breakdown_sums["accuracy"] += acc
                    breakdown_sums["x1"] += x1
                    breakdown_sums["x2"] += x2
                    breakdown_sums["x3"] += x3
                    breakdown_sumsq["format"] += fmt**2
                    breakdown_sumsq["non_repeat"] += nrep**2
                    breakdown_sumsq["accuracy"] += acc**2
                    breakdown_sumsq["x1"] += x1**2
                    breakdown_sumsq["x2"] += x2**2
                    breakdown_sumsq["x3"] += x3**2
                    for k in breakdown_counts.keys():
                        breakdown_counts[k] += 1

                    # Extra (non-reward) signals for debugging.
                    try:
                        extra_sums["x1_raw"] += float(x1_raw)
                        extra_sums["x2_raw"] += float(x2_raw)
                        extra_sums["x3_raw"] += float(x3_raw)
                        extra_sumsq["x1_raw"] += float(x1_raw) ** 2
                        extra_sumsq["x2_raw"] += float(x2_raw) ** 2
                        extra_sumsq["x3_raw"] += float(x3_raw) ** 2
                        extra_sums["query_ok"] += float(bd.get("accuracy/query_ok", 0.0))
                        extra_count += 1
                    except Exception:
                        pass

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        if want_breakdown:
            self.latest_reward_tensors = {f"accuracy_{k}": v for k, v in breakdown_tensors.items() if k in ["x1", "x2", "x3"]}
            self.latest_reward_tensors.update({f"accuracy_{k}": v for k, v in raw_tensors.items() if k in ["raw_x1", "raw_x2", "raw_x3"]})
            # also expose format/non_repeat/accuracy totals for debugging
            self.latest_reward_tensors.update({k: v for k, v in breakdown_tensors.items() if k in ["format", "non_repeat", "accuracy"]})

            self.latest_metrics = {}
            for k, s in breakdown_sums.items():
                c = max(1, breakdown_counts.get(k, 0))
                mean = float(s) / float(c)
                self.latest_metrics[f"reward_breakdown/{k}_mean"] = mean
                # Std for queue-returned (served) rewards before advantage (pop std).
                try:
                    sumsq = float(breakdown_sumsq.get(k, 0.0))
                    var = max(0.0, sumsq / float(c) - mean * mean)
                    self.latest_metrics[f"reward_breakdown/{k}_std"] = var ** 0.5
                except Exception:
                    pass
            if extra_count > 0:
                self.latest_metrics["reward_breakdown/x1_raw_mean"] = float(extra_sums["x1_raw"]) / float(extra_count)
                self.latest_metrics["reward_breakdown/x2_raw_mean"] = float(extra_sums["x2_raw"]) / float(extra_count)
                self.latest_metrics["reward_breakdown/x3_raw_mean"] = float(extra_sums["x3_raw"]) / float(extra_count)
                # Std for raw (pre-serve) rewards across samples in this batch.
                for k in ["x1_raw", "x2_raw", "x3_raw"]:
                    mean = float(extra_sums[k]) / float(extra_count)
                    var = max(0.0, float(extra_sumsq.get(k, 0.0)) / float(extra_count) - mean * mean)
                    self.latest_metrics[f"reward_breakdown/{k}_std"] = var ** 0.5
                self.latest_metrics["reward_breakdown/query_ok_rate"] = float(extra_sums["query_ok"]) / float(extra_count)
        else:
            self.latest_reward_tensors = {}
            self.latest_metrics = {}

        return reward_tensor
