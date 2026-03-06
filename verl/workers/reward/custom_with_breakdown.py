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


from collections import defaultdict

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import math_compute_score, r1v_compute_score, seg_compute_score, seg_strict_compute_score, vision_reasoner_compute_score, dr_seg_compute_score


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
        self._use_vision_reasoner_breakdown = compute_score == "vision_reasoner"
        #self._use_vision_reasoner_breakdown = None

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0
        look_box_counts = []
        component_rewards = defaultdict(list) if self._use_vision_reasoner_breakdown else None

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

            if self._use_vision_reasoner_breakdown:
                score_result = self.compute_score(response_str, ground_truth, return_breakdown=True)
                if isinstance(score_result, tuple) and len(score_result) == 2:
                    score, breakdown = score_result
                else:
                    score, breakdown = score_result, None
                if breakdown:
                    for name, value in breakdown.items():
                        component_rewards[name].append(value)
            else:
                score = self.compute_score(response_str, ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            # Collect look box counts for logging when using vision_reasoner scoring
            try:
                if self.compute_score is vision_reasoner_compute_score or self.compute_score is dr_seg_compute_score:  # Only collect breakdown for vision_reasoner or dr_seg_compute_score
                    # Lazy import to avoid cycles for other modes
                    from verl.utils.reward_score.vision_reasoner import count_look_boxes
                    cnt = count_look_boxes(response_str)
                    look_box_counts.append(cnt)
            except Exception:
                pass

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        # Expose per-batch auxiliary metrics for trainer logging
        try:
            aux_metrics = {}
            if look_box_counts:
                aux_metrics.update({
                    "vision/look_boxes_in_look/mean": float(np.mean(look_box_counts)),
                    "vision/look_boxes_in_look/max": float(np.max(look_box_counts)),
                    "vision/look_boxes_in_look/min": float(np.min(look_box_counts)),
                })
            if component_rewards:
                for name, values in component_rewards.items():
                    if not values:
                        continue
                    aux_metrics[f"reward/vision_reasoner/{name}/mean"] = float(np.mean(values))
            self.latest_metrics = aux_metrics
        except Exception:
            self.latest_metrics = {}

        return reward_tensor
