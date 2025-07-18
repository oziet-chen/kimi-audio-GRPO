# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional
import json
import logging

import torch
# deepspeed 在 trainer里
import transformers
from transformers.integrations import deepspeed
#
from datasets import load_dataset, load_from_disk
# from transformers import Qwen2VLForConditionalGeneration
# from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified, Qwen2_5OmniGRPOTrainer
from grpo.src.trainer import KimiAudioGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from IPython import embed

from math_verify import parse, verify
import numpy as np
# from jiwer import wer as jiwer_wer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    len_control: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using length reward"},
    )
    weighted_reward: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using weighted reward"},
    )
    model_type: Optional[str] = field(
        default="vl",
        metadata={"help": "Model type to use: 'vl' for Vision-Language models, 'omni' for Omni models"},
    )
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )



# === 预编译正则，提高循环内多次调用效率 ===
ANS_TAG  = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
SPAN_TAG = re.compile(r"\[(\d+\.?\d*),\s*(\d+\.?\d*)\]")
LYRIC_FILTER = re.compile(r'\[\d+:\d+.\d+\]') # HQ modify 增加一个正则预编译 用于过滤歌词中的时间信息

# no think annotation
# _FORMAT_PATTERN = re.compile(
#     r'^(?!.*<think>.*<think>)'       
#     r'(?!.*<answer>.*<answer>)'     
#     r'<think>[\s\S]+?</think>\s*<answer>[\s\S]+?</answer>$'
# )

# think annotation
_FORMAT_PATTERN = re.compile(
    r'(?!.*<answer>.*<answer>)'     
    r'<answer>[\s\S]+?</answer>$'
)


def extract_answer(text: str) -> str:
    """和你原来 extract_answer 完全等价"""
    m = ANS_TAG.search(text)
    return m.group(1).strip() if m else ""

def parse_spans(text: str) -> np.ndarray:
    """把 '[12, 19]; [24, 29]' 提取成 N×2 数组"""
    spans = SPAN_TAG.findall(text)
    if not spans:
        return np.zeros((0,2), dtype=float)
    return np.array(spans, dtype=float).reshape(-1, 2)

def iou_numpy(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    向量化版本的 compute_iou：
    intersect = sum(max(0, min(pe,ge)-max(ps,gs)))
    union = sum(len_pred)+sum(len_gt)-intersect
    """
    if pred.size==0 or gt.size==0:
        return 0.0
    inter = np.maximum(
        0.0,
        np.minimum(pred[:,None,1], gt[None,:,1])
        - np.maximum(pred[:,None,0], gt[None,:,0])
    ).sum()
    union = (pred[:,1]-pred[:,0]).sum() + (gt[:,1]-gt[:,0]).sum() - inter
    return float(inter / (union + 1e-9))


def accuracy_reward(completions, solution, **kwargs):
    """
    完整覆盖你原来所有分支，返回值 均在 [0.0, 1.0] 之间。
    """
    qtype = kwargs['problem_type'][0]
    outputs = [c[0]["content"] for c in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # 预初始化 Rouge 计算器
    rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

    for out, sol in zip(outputs, solution):
        pred_ans = extract_answer(out)
        gt_ans   = extract_answer(sol)

        if qtype == "multiple choice":
            # 完全一致 =1，否则 0
            reward = 1.0 if pred_ans == gt_ans else 0.0

        elif qtype == "numerical":
            # 四舍五入到小数后2位再比较
            try:
                p = round(float(pred_ans.replace(',','')), 2)
                g = round(float(gt_ans.replace(',','')), 2)
                reward = 1.0 if p == g else 0.0
            except:
                reward = 0.0

        elif qtype == "OCR":
            # 1 - WER，clip 到 [0,1]
            err = jiwer_wer(gt_ans, pred_ans)
            reward = max(0.0, min(1.0, 1 - err))

        elif qtype == "free-form":
            # # 取 rougeL f-measure 的平均（和你 compute_rouge_score 类似）
            scores = rouge.score(gt_ans, pred_ans)
            # # 平均 rouge1/2/L 或者单独用 rougeL
            reward = max(0.0, min(1.0, scores['rougeL'].fmeasure))

        elif qtype == "regression":
            # 1 - 相对误差，clip 到 [0,1]
            try:
                p = float(pred_ans.replace(',',''))
                g = float(gt_ans.replace(',',''))
                rel_err = abs(p - g)/(abs(g)+1e-9)
                reward = max(0.0, min(1.0, 1 - rel_err))
            except:
                reward = 0.0

        elif qtype == "temporal retrieval":
            # 多段 IOU
            p_spans = parse_spans(pred_ans)
            g_spans = parse_spans(gt_ans)
            reward = iou_numpy(p_spans, g_spans)

        elif qtype == "counting":
            try:
                # 尝试 symbolic verification
                answer = parse(pred_ans)
                reward = 1.0 if float(verify(answer, parse(gt_ans))) > 0 else 0.0
            except:
                reward = 1.0 if pred_ans.strip() == gt_ans.strip() else 0.0

        else:
            reward = 0.0

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {out}\n")
                f.write(f"Solution: {sol}\n")
            

    return rewards


def format_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        text = c[0]["content"].strip()          # <-- remove surrounding whitespace
        rewards.append(
            1.0 if _FORMAT_PATTERN.fullmatch(text) else 0.0
        )
    return rewards



reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )

# Common format
# QUESTION_TEMPLATE = (
#     "{Question}\n"
#     "Please think about this question as if you were a human pondering deeply. "  # no think annotation
#     # "Make sure to carefully consider both the visual and audio information before answering. "
#     "Make sure to carefully consider the audio information before answering. " # HQ modify
#     "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "  # no think annotation
#     "It's encouraged to include self-reflection or verification in the reasoning process. "  # no think annotation
#     "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."  # no think annotation
#     # "Just give your final answer between the <answer> </answer> tags." # think annotation
# )




# TYPE_TEMPLATE = {
#     # "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
#     "multiple choice": " Please provide only the content of a single option within the <answer> </answer> tag (for example, consistent with one of the options).", # HQ modify
#     "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
#     "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
#     "free-form": " Please provide your text answer within the <answer> </answer> tags.",
#     "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
#     "temporal retrieval": (
#         " The query describes an event in the video. Identify all time spans during which it occurs.\n"
#         "List them inside <answer>...</answer>, e.g., <answer>[12.0, 19.0]; [24.0, 29.0]</answer>."
#     ),
#     "counting": " Please provide the numerical value (e.g., 1 or 7) in <answer> </answer> tags.",
# }

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
        # HQ modify 如果不行 加下这个参数 dataset = Dataset.from_json("data.jsonl", lines=True)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # 选择合适的训练器类
    if script_args.model_type.lower() == "kimi":
        print("Using KimiAudioGRPOTrainer for Kimi-Audio model")
        trainer_cls = KimiAudioGRPOTrainer
    else:
        raise
    #     print("Using standard VL GRPO Trainer")
    #     trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    
    print("Using trainer class: ", trainer_cls)

    # 初始化GRPO训练器
    trainer_kwargs = {
        "model": model_args.model_name_or_path,
        "reward_funcs": reward_funcs,
        "args": training_args,
        "script_args": script_args,
        "train_dataset": dataset[script_args.dataset_train_split],
        "eval_dataset": dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        "peft_config": get_peft_config(model_args),
        "attn_implementation": model_args.attn_implementation,
    }
    
    # 对于Omni模型添加额外参数
    # if script_args.model_type.lower() == "omni":
    #     trainer_kwargs["use_audio_in_video"] = script_args.use_audio_in_video
    
    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Echoink的输入 参考 config_info.md
    # print("\t====>script_args =>\n",script_args)
    # print("\t====>training_args =>\n",training_args)
    # print("\t====>model_args =>\n",model_args)


    main(script_args, training_args, model_args)