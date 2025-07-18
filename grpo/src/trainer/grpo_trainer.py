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
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoConfig,  # 添加这一行
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url


import copy
from IPython import embed
import re

## 
from finetune_codes.model import KimiAudioModel
from finetune_codes.datasets import LazySupervisedDataset
from kimia_infer.api.prompt_manager import KimiAPromptManager


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    
import torch.distributed as dist
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class KimiAudioGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method with Qwen2.5Omni model support.
    This extends the GRPO algorithm to handle multimodal inputs including audio and video.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        use_audio_in_video: bool = True,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

        # HQ grpo: 参考SFT 增加了一个参数 low_cpu_mem_usage
        model_init_kwargs["low_cpu_mem_usage"] = not is_deepspeed_zero3_enabled()

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            
            # 在 Qwen2_5OmniGRPOTrainer.__init__ 方法中的模型初始化部分
            if "Kimi-Audio" in model_id:
                # 为 Qwen2.5-Omni 模型创建一个新的 model_init_kwargs 字典
                omni_model_init_kwargs = model_init_kwargs.copy()
                if "use_cache" in omni_model_init_kwargs:
                    del omni_model_init_kwargs["use_cache"]
                
                # 加载配置并修改

                # config = AutoConfig.from_pretrained(model_id)
                # config.enable_audio_output = False  # 在配置中禁用音频输出
                
                # 使用修改后的配置加载模型
                model = KimiAudioModel.from_pretrained(
                    model_id, 
                    device_map=None,
                    trust_remote_code=True, # 增加该参数 避开hf警告
                    **omni_model_init_kwargs
                )
                # model = KimiAudioModel.init_from_pretrained(
                #     model_id, 
                #     omni_model_init_kwargs
                # )
                # HQ grpo: Kimi-audio的config 通过模型获取 但我们并不需要修改其中的配置(由于用不上 就注掉了 用的话 直接从model读)
                # config = model.config

                # 确保 talker 被禁用
                # model.disable_talker()
                # print("Talker component has been disabled")
            else:
                raise ValueError(f"Unsupported model ID for Qwen2.5OmniGRPOTrainer: {model_id}")
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # 在创建参考模型部分
        if is_deepspeed_zero3_enabled():
            if "Kimi-Audio" in model_id:
                # 同样处理参考模型
                omni_model_init_kwargs = model_init_kwargs.copy()
                if "use_cache" in omni_model_init_kwargs:
                    del omni_model_init_kwargs["use_cache"]
                
                # config = AutoConfig.from_pretrained(model_id)
                # config.enable_audio_output = False  # 在配置中禁用音频输出
                
                self.ref_model = KimiAudioModel.from_pretrained(
                    model_id, 
                    device_map=None,
                    trust_remote_code=True,
                    **omni_model_init_kwargs
                )
                # self.ref_model = KimiAudioModel.init_from_pretrained(
                #     model_id, 
                #     omni_model_init_kwargs
                # )
                # HQ grpo: ref这里和前面一样 也是加载模型后 取config(由于用不上 就注掉了)
                #  config = self.ref_model.config

                # self.ref_model.disable_talker()
                # print("Reference model talker component has been disabled")
            else:
                raise ValueError(f"Unsupported model ID for reference model: {model_id}")
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

       
        # if processing_class is None:
        #     if "Kimi-Audio" in model_id:
        #         processing_class = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        #         pad_token_id = processing_class.tokenizer.pad_token_id
        #         processing_class.pad_token_id = pad_token_id
        #         processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        #     else:
        #         raise ValueError(f"Unsupported model ID for processor: {model_id}")

        # HQ grpo: EchoInk这里 是载入了一个processing_class 就是一个自定义的tokenizer 我们看Kimi-audio的SFT 实际上并不需要这个东西，tokenizer是在自定义数据集类中起的作用
        # HQ grpo:     而在EchoInk的trainer中实际上它的作用涉及4处：
        # HQ grpo:     apply_chat_template
        # HQ grpo:     接收原始输入 进行token化和padding
        # HQ grpo:     读取 eos_token_id 进行判断
        # HQ grpo:     batch_decode
        # HQ grpo:     这在Kimi-audio中 是可以通过 text_tokenizer来实现 实际上这被封装到了 KimiAPromptManager 类中 上面的几个点 我们均通过这个类的实例化对象来实现
        # HQ grpo:     补充修改：在让model支持generate的时候 需要使用 prompt_manager，因此把这个对象 需要放到model中
        self.prompt_manager = KimiAPromptManager(
            model_path=model_id, kimia_token_offset=model.config.kimia_token_offset, kimia_text_audiodelaytokens=model.config.kimia_mimo_audiodelaytokens
        )
        self.ref_prompt_manager = KimiAPromptManager(
            model_path=model_id, kimia_token_offset=self.ref_model.config.kimia_token_offset, kimia_text_audiodelaytokens=self.ref_model.config.kimia_mimo_audiodelaytokens
        )
        # HQ grpo: 另外 我们去掉了trainer中 对 use_audio_in_video 和 max_prompt_length 参数的支持，第一个参数不需要 第二个参数是对token的截取，而kimi-audio的输入已经是token化了的

        # HQ grpo: 奖励函数这里我们保持不动
        # The rest of the initialization is the same as in Qwen2VLGRPOTrainer
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        # self.max_prompt_length = args.max_prompt_length # HQ grpo: 前文已阐明
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.weighted_reward = script_args.weighted_reward

        # HQ grpo: Echoink使用了GenerationConfig进行批量采样 但类似的实现在Kimi-audio中并不支持
        # HQ grpo:   我们需要在model.py中 参考 kimia.py的推理 自己实现 一个支持采样的 generate 方法
        # HQ grpo:   这里的generation_config仅做参数存储器
        # HQ grpo:   Kimi-audio支持的采样 不支持全部采样参数
        self.sampling_params = {
            "audio_temperature": 1.0,
            "audio_top_k": 10,
            "text_temperature": 1.0,
            "text_top_k": 10,
            # "text_top_p": 0.95,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }
        # self.generation_config = GenerationConfig(
        #     max_new_tokens=self.max_completion_length,
        #     do_sample=True,
        #     top_p=0.95,   
        #     temperature=1, # HACK
        #     num_return_sequences=self.num_generations,
        #     pad_token_id=pad_token_id,
        # )
        # HQ grpo: 其他几个采样代码用不到 就删除了

        #
        self.len_control = script_args.len_control
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)


        # HQ grpo: 参考SFT的代码，对传入的ds 进行一次转换处理，之所以放到这里 是因为 要用到 text_tokenizer
        # HQ grpo:     但是这里代码其实是有缺 比如 传入的whisper_model是要参与训练的 代码里虽然传进去了 但是完全没用到
        dataset_cls = LazySupervisedDataset
        train_dataset = dataset_cls(train_dataset, whisper_model=model.whisper_model, text_tokenizer=self.prompt_manager.text_tokenizer, max_len=script_args.model_max_length, kimia_token_offset=self.prompt_manager.kimia_token_offset)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        model.config_prompt_manager(self.prompt_manager, model.config)
        
        self.ref_model.config_prompt_manager(self.ref_prompt_manager, self.ref_model.config)
        
        # HQ grpo: 注意要删除zero3.json中的optimizer和scheduler

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    # HQ grpo: 我们的数据是已经预处理后的 因此不需要重写这个方法
    # HQ grpo:     这里 我们按照数据集返回的 有效列 进行返回 而不是删除
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["input_ids","text_input_ids","whisper_input_feature","is_continuous_mask"]

    def _get_per_token_logps(self, model, input_ids, text_input_ids, **kwargs):
        # local_rank = dist.get_rank()
        
        # print(
        #     "_get_per_token_logps doing1 at rank = ", 
        #     local_rank, 
        #     " input_ids.size",input_ids.size(), 
        #     " text_input_ids.size",text_input_ids.size(), 
        #     " whisper_input_feature.size",kwargs["whisper_input_feature"].size(),
        #     " is_continuous_mask.size",kwargs["is_continuous_mask"].size(),
        #     " prompt_length",kwargs["prompt_length"]
        # )
#         _get_per_token_logps doing1 at rank =  3  input_ids.size torch.Size([4, 1180])  text_input_ids.size torch.Size([4, 1180])  whisper_input_feature.size torch.Size([4, 960000])  is_continuous_mask.size torch.Size([4, 1180])  prompt_length 1167
# _get_per_token_logps doing1 at rank =  1  input_ids.size torch.Size([4, 1425])  text_input_ids.size torch.Size([4, 1425])  whisper_input_feature.size torch.Size([4, 960000])  is_continuous_mask.size torch.Size([4, 1425])  prompt_length 1410
# _get_per_token_logps doing1 at rank =  2  input_ids.size torch.Size([4, 1421])  text_input_ids.size torch.Size([4, 1421])  whisper_input_feature.size torch.Size([4, 960000])  is_continuous_mask.size torch.Size([4, 1421])  prompt_length 1404

        # HQ grpo: 避免去修改modeling_kimia.py

        # 使用循环 以支持batch
        logits_all = []
        for index in range(input_ids.size()[0]):
            logits = model._generate_loop_for_get_per_token_logits(
                audio_input_ids=input_ids[index:(index + 1),:],
                text_input_ids=text_input_ids[index:(index + 1),:], 
                continous_feature=kwargs["whisper_input_feature"],
                is_continuous_mask=kwargs["is_continuous_mask"][index:(index + 1),:],
                # prompt_length=kwargs["prompt_length"],
            ) # 只取 text_logits
            logits_all.append(logits)
        logits = torch.cat(logits_all, dim=0).requires_grad_(True) # 这里是自己拼接的张量 需要设置该张量支持梯度计算！！！

        # batch 不被支持
        # logits = model._generate_loop_for_get_per_token_logits(
        #         audio_input_ids=input_ids[index:(index + 1),:],
        #         text_input_ids=text_input_ids[index:(index + 1),:], 
        #         continous_feature=kwargs["whisper_input_feature"],
        #         is_continuous_mask=kwargs["is_continuous_mask"][index:(index + 1),:],
        #         # prompt_length=kwargs["prompt_length"],
        #     ) # 只取 text_logits
        # print("_get_per_token_logps doing2 at ", logits.size(), text_input_ids.size(), local_rank)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        # test = 1 # _generate_loop_for_get_per_token_logits中也有设置
        # input_ids = text_input_ids[:test, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # input_ids = text_input_ids[:test, 1:]
        input_ids = text_input_ids[:, 1:]
        # print("_get_per_token_logps doing3 at ", logits.size(), input_ids.size(), local_rank)
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)

        return torch.stack(per_token_logps)
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # 把会话传过来
        prompts = [x["prompt"] for x in inputs]

        # HQ grpo: 组织输入 输入的是batch 但由于只支持batch=1 因此实际输入就是0下标对应的元素
        input_copy = copy.deepcopy(inputs[0])
        
        # 原始input的其他key 计算奖励的时候 还要用 因此copy一个新的对象 便于操作；这里删除非参数key
        del input_copy["mid"]
        del input_copy["solution"]
        del input_copy["problem_type"]
        del input_copy["prompt"]

        # 有效Key
        # input_ids=audio_input_ids,
        # text_input_ids=text_input_ids,
        # whisper_input_feature=audio_features,
        # is_continuous_mask=is_continuous_mask,

        prompt_inputs = input_copy
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids_audio = prompt_inputs["input_ids"]
        prompt_ids_text = prompt_inputs["text_input_ids"]
        prompt_is_continuous_mask = prompt_inputs["is_continuous_mask"]

        # Generate completions
        prompt_completion_ids_audio = [] # 我们计算各个生成序列token 在目标模型上的logits 因为只有text输出的logits是我们使用的（参与计算） 因此这里音频输出的部分 可以忽略（存疑？） 我们使用pad填充一个等长的张量备用（？）
        # 这个变量 仅做打印使用
        prompt_completion_ids_text = []
        prompt_completions_text = []
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # 为Qwen2.5Omni模型生成文本和音频
            # HQ grpo: 为model 实现 generate方法 由于不支持 generation_config ，这里需要使用for循环
            for _ in range(self.num_generations): # 
                # print("我在采样推理答案")
                _, generated_text, generated_wav_tokens, generated_text_tokens = unwrapped_model.generate_with_tokens(**prompt_inputs, **self.sampling_params, output_type="text", max_new_tokens = self.max_completion_length)

                prompt_completion_ids_audio.append(generated_wav_tokens)
                generated_text_tokens.append(self.prompt_manager.extra_tokens.kimia_text_eos) # HQ 防止序列为空 导致的后续计算 completion_mask 报错
                prompt_completion_ids_text.append(torch.tensor(generated_text_tokens))
                prompt_completions_text.append(generated_text)
            
            prompt_length = prompt_ids_audio.size(1)
            # print("采样推理答案的输入指令长度 ", prompt_length)
            # prompt_ids = prompt_completion_ids[:, :prompt_length]
            # completion_ids = prompt_completion_ids[:, prompt_length:]
            # prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
        print("\t", len(prompt_completions_text), len(prompt_completions_text[0]))
        # print("看下推理结果 prompt_completion_ids_audio：",prompt_completion_ids_audio, type(prompt_completion_ids_audio)) 
        # # 这里返回的audio token都是空的。。。
        # print("看下推理结果 prompt_completion_ids_text：",prompt_completion_ids_text, type(prompt_completion_ids_text))
        # local_rank = dist.get_rank()
        # print("看下推理结果 prompt_completions_text：",prompt_completions_text, type(prompt_completions_text), " rank=",local_rank )

        # size不同 不能直接用stack
        # prompt_completion_ids_text = torch.stack(prompt_completion_ids_text, dim=0)

        device = self.accelerator.device
        
        prompt_completion_ids_text = pad_sequence(
            prompt_completion_ids_text, 
            batch_first=True,  
            padding_value=model.extra_tokens.pad,
        )
        
        # print("看下推理结果 prompt_completion_ids_text：", prompt_completion_ids_text.size(), prompt_completion_ids_text.device, prompt_ids_text.device, device)

        if prompt_completion_ids_text.device != device: # HQ
            prompt_completion_ids_text = prompt_completion_ids_text.to(device)
        
        # Mask everything after the first EOS token
        is_eos = prompt_completion_ids_text == self.prompt_manager.extra_tokens.kimia_text_eos # HQ

        # 输入数据中所有序列均未包含结束标记（EOS） 接下来的代码会报错
        if is_eos.device != device: # HQ
            is_eos = is_eos.to(device)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        # print("completion_mask is_eos.size() => ",is_eos.size(), is_eos.any(dim=1), is_eos)
        # completion_mask is_eos.size() =>  torch.Size([4, 99]) tensor([False, False, False, False], device='cuda:3') 
        # if is_eos.size()[1] > 0:
            # print("normal dealing...")
            # is_eos.int().argmax(dim=1)
            # [rank1]: IndexError: argmax(): Expected reduction dim 1 to have non-zero size.
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # print("eos_idx ", eos_idx.size(),eos_idx) # eos_idx  torch.Size([4]) tensor([99, 99, 99, 99], device='cuda:3')
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # print(completion_mask.size(), completion_mask)
        
        # HQ grpo: input_ids text_input_ids 单独传了 这里要移除相应的key
        # print("type(prompt_inputs[whisper_input_feature]) ====> ",type(prompt_inputs["whisper_input_feature"]),prompt_inputs["whisper_input_feature"])
        for key in ["input_ids", "text_input_ids", "is_continuous_mask"]:
            if key in prompt_inputs:
                prompt_inputs.pop(key)
                
        repeat_factor = len(prompt_completion_ids_text)
        
        # HQ grpo: 重复不变的信息
        if "whisper_input_feature" in prompt_inputs:
            # prompt_inputs["whisper_input_feature"] = torch.from_numpy(np.array(prompt_inputs["whisper_input_feature"])).repeat_interleave(repeat_factor, dim=0)
            prompt_inputs["whisper_input_feature"] = torch.from_numpy(np.array(prompt_inputs["whisper_input_feature"]))
        
        # print("看下 prompt_inputs[whisper_input_feature] 批次处理后的 size ", prompt_inputs["whisper_input_feature"].size()) # torch.Size([4, 960000])
        
        prompt_completion_ids_audio = torch.full(
            prompt_completion_ids_text.size(), 
            fill_value=model.extra_tokens.pad, # HQ ? 应该填充0 还是pad
            dtype=prompt_completion_ids_text.dtype,
            device=prompt_completion_ids_text.device
        )
        prompt_completion_is_continuous_mask = torch.full(
            prompt_completion_ids_text.size(), 
            fill_value=False,
            dtype=prompt_completion_ids_text.dtype,
            device=prompt_completion_ids_text.device
        )

        # 输入只有一个 我们要向 num_generations 扩充第一维 对应batch
        prompt_ids_text = torch.stack([prompt_ids_text for _ in range(self.num_generations)], dim=0)
        prompt_ids_audio = torch.stack([prompt_ids_audio for _ in range(self.num_generations)], dim=0)
        prompt_is_continuous_mask = torch.stack([prompt_is_continuous_mask for _ in range(self.num_generations)], dim=0)

        # 原始可用输入是1xN的 我们要向 num_generations 扩充为3维
        prompt_completion_ids_text = torch.unsqueeze(prompt_completion_ids_text, 1)
        prompt_completion_ids_audio = torch.unsqueeze(prompt_completion_ids_audio, 1)
        prompt_completion_is_continuous_mask = torch.unsqueeze(prompt_completion_is_continuous_mask, 1)

        # 原始输入和输出的id进行拼接
        prompt_ids_text = torch.cat((prompt_ids_text, prompt_completion_ids_text), dim=-1)
        prompt_ids_audio = torch.cat((prompt_ids_audio, prompt_completion_ids_audio), dim=-1)
        prompt_inputs["is_continuous_mask"] = torch.cat((prompt_is_continuous_mask, prompt_completion_is_continuous_mask), dim=-1)
        # test
        prompt_ids_text = torch.squeeze(prompt_ids_text)
        prompt_ids_audio = torch.squeeze(prompt_ids_audio)
        # prompt_inputs["whisper_input_feature"] = torch.squeeze(prompt_inputs["whisper_input_feature"]).unsqueeze(1)
        prompt_inputs["is_continuous_mask"] = torch.squeeze(prompt_inputs["is_continuous_mask"])

        # attention_mask = (prompt_ids_text != model.extra_tokens.pad).type(torch.int32)
        # prompt_inputs["attention_mask"] = attention_mask
        # print('prompt_inputs["attention_mask"] size = ', prompt_inputs["attention_mask"].size())

        
        # prompt_inputs["prompt_length"] = torch.tensor([[prompt_length] * self.num_generations]).unsqueeze(1)
        prompt_inputs["prompt_length"] = prompt_length
        # print("prompt_inputs[prompt_length] size => ", prompt_inputs["prompt_length"] )
        
        per_token_logps = self._get_per_token_logps(model, prompt_ids_audio, prompt_ids_text, **prompt_inputs)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        # print("计算loss 1")
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_ids_audio, prompt_ids_text, **prompt_inputs)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_ids_audio, prompt_ids_text, **prompt_inputs)
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
        # print("计算loss 2")
        # 计算KL散度
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        # print("计算loss 3")
        # HQ grpo: 解码生成的完成
        completions = prompt_completions_text
        if is_conversational(inputs[0]):
            # 把message_type 加上
            completions = [[{"role": "assistant", "message_type":"text", "content": completion}] for completion in completions]
            print("prompts = \n",prompts)
            print("completions = \n",completions)
        # HQ grpo: 后面的代码基本没动
        # print("计算loss 4")
        # 计算奖励
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        # print("计算loss 5")
        # === 总奖励 START ===
        if not self.weighted_reward:
            rewards = rewards_per_func.sum(dim=1)
        else:
            # [accuracy reward, format reward]
            reward_weights = torch.tensor([2.0, 1.0], dtype=torch.float32, device=device) 
            rewards = (rewards_per_func * reward_weights).sum(dim=1)

    
        # 长度控制奖励
        if self.len_control:
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
                    
            if len(selected_indices) > 1:     
                for idx in selected_indices:
                    # if 320 <= lenth_list[idx] <= 512:
                    if 160 <= lenth_list[idx] <= 256:
                        rewards[idx] += 0.2

        # 分组奖励计算
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # 奖励归一化
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # 损失计算
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        
        # With KL loss
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        

        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # print("计算loss 6", loss)
        # 记录指标
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        
        # num_devices = gathered_rewards.size(0) // self.num_generations 
        # rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        rewards_per_device, num_devices = self.safe_group_rewards(gathered_rewards, self.num_generations, mode="pad")
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        
        # if len([p.grad for p in model.parameters() if p.grad is None]):
        #     print("计算loss ", loss," error!")
        # else:
        #     print("计算loss ", loss," over!")

        return loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()


    def safe_group_rewards(self, gathered_rewards: torch.Tensor, num_generations: int, mode: str = "pad"):
        total_samples = gathered_rewards.shape[0]
        remainder = total_samples % num_generations

        if remainder != 0:
            if mode == "pad":
                pad_size = num_generations - remainder
                pad_values = torch.zeros(pad_size, device=gathered_rewards.device, dtype=gathered_rewards.dtype)
                gathered_rewards = torch.cat([gathered_rewards, pad_values], dim=0)
                print(f"[safe_group_rewards] Warning: padded {pad_size} samples to match num_generations")
            elif mode == "truncate":
                gathered_rewards = gathered_rewards[: total_samples - remainder]
                print(f"[safe_group_rewards] Warning: truncated {remainder} extra samples to match num_generations")
            else:
                raise ValueError(f"Unsupported mode: {mode}. Use 'pad' or 'truncate'.")

        num_devices = gathered_rewards.size(0) // num_generations
        reshaped_rewards = gathered_rewards.view(num_devices, num_generations)
        return reshaped_rewards, num_devices
