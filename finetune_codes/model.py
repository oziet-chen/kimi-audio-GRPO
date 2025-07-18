import os
import argparse
from typing import Optional, List
import shutil
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from .modeling_kimia import MoonshotKimiaForCausalLM

from kimia_infer.utils.sampler import KimiASampler
from kimia_infer.api.prompt_manager import KimiAPromptManager

import torch.distributed as dist


class KimiAudioModel(MoonshotKimiaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.whisper_model = WhisperEncoder("openai/whisper-large-v3", mel_batch_size=20, unfreeze_online_whisper_model=True)

        self.detokenizer = None
    
    # HQ grpo: 为创建好的model 配置一个prompt_manager
    def config_prompt_manager(self, prompt_manager, model_config):
        self.prompt_manager = prompt_manager  

        self.extra_tokens = self.prompt_manager.extra_tokens
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]
        
        self.kimia_text_audiodelaytokens = model_config.kimia_mimo_audiodelaytokens
        self.kimia_token_offset = model_config.kimia_token_offset

        print("self._name_or_path ==============>", self.name_or_path)
        self.whisper_model = WhisperEncoder(os.path.join(self.name_or_path, "whisper-large-v3"), mel_batch_size=20, unfreeze_online_whisper_model=True)

        pretrained_state_dict = self.state_dict()
        for n, p in self.whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p
        self.load_state_dict(pretrained_state_dict)
        

    @classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs):
        
        if os.path.exists(model_name_or_path):
            # local path
            cache_path = model_name_or_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_name_or_path)

        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path, 
            device_map=None,
            torch_dtype=torch.bfloat16, trust_remote_code=True, **model_load_kwargs,
        )

        whisper_model = WhisperEncoder(
            os.path.join(cache_path, "whisper-large-v3"), mel_batch_size=20, unfreeze_online_whisper_model=True
        )
        kimia_model = cls(audio_model.config)

        # merge audio model and whisper model's state dict
        pretrained_state_dict = audio_model.state_dict()
        
        for n, p in whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p

        kimia_model.load_state_dict(pretrained_state_dict)

        return kimia_model


    
    @staticmethod
    def export_model(input_dir, output_dir):
        print("Loading model from {}".format(input_dir))
        kimiaudio = KimiAudioModel.from_pretrained(input_dir)

        print("Saving Kimi-Audio LM to {}".format(output_dir))
        audio_model = MoonshotKimiaForCausalLM(kimiaudio.config)
        audio_model_state_dict = {k: v for k, v in kimiaudio.state_dict().items() if not k.startswith("whisper_model")}
        audio_model.load_state_dict(audio_model_state_dict)

        audio_model.save_pretrained(output_dir)

        shutil.copyfile("finetune_codes/configuration_moonshot_kimia.py", os.path.join(output_dir, "configuration_moonshot_kimia.py"))
        shutil.copyfile("finetune_codes/modeling_kimia.py", os.path.join(output_dir, "modeling_moonshot_kimia.py"))

        from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperModel

        whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")

        kimiaudio_whisper_encoder_state_dict = {k.replace("speech_encoder.", "encoder."): v for k, v in kimiaudio.whisper_model.state_dict().items() if k.startswith("speech_encoder")}

        missing_keys, unexpected_keys = whisper_model.load_state_dict(kimiaudio_whisper_encoder_state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

        for k in missing_keys:
            assert k.startswith("decoder"), f"Missing keys: {k}"

        whisper_model.save_pretrained(os.path.join(output_dir, "whisper-large-v3"))

        print("Exported Kimi-Audio LM and Whisper model to {}".format(output_dir))
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if type(whisper_input_feature) == type(None): # HQ grpo
            whisper_feats = whisper_input_feature
        else:
            if type(whisper_input_feature[0]) == torch.Tensor:
                whisper_input_feats = whisper_input_feature[0].unsqueeze(0)[:, :].to(torch.cuda.current_device())
            else:
                whisper_input_feats = torch.from_numpy(whisper_input_feature[0]).unsqueeze(0)[:, :].to(torch.cuda.current_device())

            # assert input_ids.size()[0] == 1
            # whisper_input_feats.size() init =  torch.Size([1, 960000]) torch.Size([4, 1485])
            # whisper_input_feats before size =  torch.Size([1, 960000])
            # whisper_input_feats after size =  torch.Size([4, 960000])
            # whisper_input_feats.size() prepare =  torch.Size([4, 960000])
            # whisper_feats infer size =  torch.Size([1, 3000, 1280])
            # whisper_feats reshape size =  torch.Size([1, 750, 5120]) 理论上 应该是 4 750 5120
            whisper_feats = self.whisper_model(whisper_input_feats)
            whisper_feats = whisper_feats.reshape(
                whisper_feats.shape[0],
                int(whisper_feats.shape[1] // 4),
                whisper_feats.shape[2] * 4,
            )
            # print("whisper_feats reshape size = ", whisper_feats.size())
            if whisper_feats.size()[0] != input_ids.size()[0]:
                whisper_feats = whisper_feats.repeat_interleave(input_ids.size()[0], dim=0) # HQ                 
            # print("whisper_feats repeat size = ", whisper_feats.size()) # 还是不支持批量 
        return super().forward(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_feats,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generation_mode=generation_mode,
            return_dict=return_dict,
        )

    # def get_next_token(self, logits):
    #     if len(logits.shape) == 3:
    #         logits = logits[:, -1]
    #     logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
    #     next_token = torch.argmax(logprobs, dim=-1)
    #     return next_token

    @torch.inference_mode() 
    def _generate_loop_for_get_per_token_logits(
        self,
        audio_input_ids: torch.Tensor,  # input audio tokens
        text_input_ids: torch.Tensor = None,  # input text tokens if use multi-input
        prompt_length: int = 50,
        is_continuous_mask: torch.Tensor = None,
        continous_feature: torch.Tensor = None,
        output_type: str = "text",
    ):
        assert output_type == "text"

        # HQ 感觉这里的推理 不支持批量 先只取多个候选中的1个 进行调试

        # test = 1
        # _get_per_token_logps doing1 at  3 torch.Size([4, 1180]) torch.Size([4, 1180]) dict_keys(['whisper_input_feature', 'is_continuous_mask', 'prompt_length'])
        # decoder_input_audio_ids = audio_input_ids[:test,:].clone()
        # decoder_input_text_ids = text_input_ids[:test,:].clone()
        decoder_input_audio_ids = audio_input_ids[:,:].clone()
        decoder_input_text_ids = text_input_ids[:,:].clone()
        decoder_position_ids = (
            torch.arange(
                0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device()
            )
            .unsqueeze(0)
            .long()
        )
    
        decoder_input_whisper_feature = continous_feature # 不变的值 改预处理后的批量
        # decoder_is_continuous_mask = is_continuous_mask[:test,:]
        decoder_is_continuous_mask = is_continuous_mask[:,:]


        # print(" *decoder_input_whisper_feature = ",len(decoder_input_whisper_feature), type(decoder_input_whisper_feature), decoder_input_whisper_feature[0].shape)
        # print(" *decoder_is_continuous_mask = ",decoder_is_continuous_mask.size())
        
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1
        
        text_logit_all = []
        max_new_tokens = 1 # audio_input_ids.shape[1] - prompt_length
        local_rank = dist.get_rank()
        for i in range(max_new_tokens):
            # print("\t\t last_position_id = ",last_position_id, " at ", i , "/", max_new_tokens,  " rank=", local_rank, decoder_input_audio_ids.size())
            # HQ grpo: 替换 alm.forward 为model
            _, text_logits, past_key_values = self(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )
            text_logit_all.append(text_logits)
            # print("\t\t i = ",i, "/", max_new_tokens,  " rank=", local_rank, text_logits.size(), len(text_logit_all))

            # decoder_input_audio_ids = audio_input_ids[:,last_position_id:(last_position_id + 1)].clone() #.unsqueeze(1)
            # decoder_input_text_ids = text_input_ids[:,last_position_id:(last_position_id + 1)].clone() #.unsqueeze(1)

            # decoder_position_ids = (
            #     torch.zeros(1, 1, device=torch.cuda.current_device())
            #     .fill_(last_position_id + 1)
            #     .long()
            #     .view(1, 1)
            # )
            # last_position_id += 1
            
            # decoder_input_whisper_feature = None
            # decoder_is_continuous_mask = None

        result = torch.cat(text_logit_all, dim=1)
        # print("result.size() = ", result.size())
        return result
        
    @torch.inference_mode()
    def _generate_loop(
        self,
        audio_input_ids: torch.Tensor,  # input audio tokens
        text_input_ids: torch.Tensor = None,  # input text tokens if use multi-input
        max_new_tokens: int = 50,
        audio_top_k: int = 5,
        audio_temperature: float = 0.0,
        audio_repetition_penalty: float = 1.0,
        audio_repetition_window_size: int = 64,
        text_top_k: int = 5,
        # text_top_p: float = 0.0,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: torch.Tensor = None,
        continous_feature: torch.Tensor = None,
        output_type: str = "text",
    ):

        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            # text_top_p=text_top_p,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        text_stream_is_finished = False

        # HQ grpo: 这个值感觉需要调大，暂时不动
        previous_audio_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )
        text_previous_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        decoder_position_ids = (
            torch.arange(
                0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device()
            )
            .unsqueeze(0)
            .long()
        )
        decoder_input_whisper_feature = continous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        # print("decoder_input_whisper_feature = ",len(decoder_input_whisper_feature), type(decoder_input_whisper_feature), decoder_input_whisper_feature[0].shape)
        # print("decoder_is_continuous_mask = ",decoder_is_continuous_mask.size())

        
        last_position_id = decoder_input_audio_ids.shape[1] - 1

        valid_text_length = 0
        valid_audio_length = 0

        # local_rank = dist.get_rank()
        # HQ grpo: 删除tqdm
        for i in range(max_new_tokens):
            # print("\t\t\t_generate_loop last_position_id = ",last_position_id, " at ", i , "/", max_new_tokens,  " rank=", local_rank)
            # HQ grpo: 替换 alm.forward 为model
            audio_logits, text_logits, past_key_values = self(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )

            # Sample text token using the sampler
            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
            )

            # Sample audio token using the sampler
            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
            )

            if text_stream_is_finished:
                next_token_text.fill_(self.extra_tokens.kimia_text_blank)
            elif next_token_text.item() == self.extra_tokens.kimia_text_eos:
                text_stream_is_finished = True
            else:
                valid_text_length += 1

            text_previous_tokens[i : i + 1] = next_token_text

            if i < self.kimia_text_audiodelaytokens:
                next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
            else:
                if output_type == "text":
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    valid_audio_length += 1

            previous_audio_tokens[i : i + 1] = next_audio_token

            audio_stream_is_finished = next_audio_token.item() in self.eod_ids

            if (
                output_type == "text"
                and text_stream_is_finished
                or output_type == "both"
                and audio_stream_is_finished
            ):
                return_text_tokens = (
                    text_previous_tokens[:valid_text_length]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return_audio_tokens = (
                    previous_audio_tokens[
                        self.kimia_text_audiodelaytokens : valid_audio_length
                        + self.kimia_text_audiodelaytokens
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return return_audio_tokens, return_text_tokens
            else:
                decoder_input_audio_ids = next_audio_token.unsqueeze(1)
                decoder_input_text_ids = next_token_text.unsqueeze(1)
                # print("decoder_input_audio_ids size=>", decoder_input_audio_ids.size(),decoder_input_audio_ids)
                # print("\tdecoder_input_text_ids size=>", decoder_input_text_ids.size(),decoder_input_text_ids)
                # decoder_input_audio_ids size=> torch.Size([1, 1]) tensor([[151666]], device='cuda:2')
                # decoder_input_text_ids size=> torch.Size([1, 1]) tensor([[29]], device='cuda:2')
                decoder_position_ids = (
                    torch.zeros(1, 1, device=torch.cuda.current_device())
                    .fill_(last_position_id + 1)
                    .long()
                    .view(1, 1)
                )
                last_position_id += 1
                # print("=========================> ERROR!!!")
                decoder_input_whisper_feature = None
                decoder_is_continuous_mask = None

                # print("decoder_input_audio_ids =>", decoder_input_audio_ids) # tensor([[151666]]

        return_text_tokens = (
            text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
        )
        return_audio_tokens = (
            previous_audio_tokens[
                self.kimia_text_audiodelaytokens : valid_audio_length
                + self.kimia_text_audiodelaytokens
            ]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        
        return return_audio_tokens, return_text_tokens

    @torch.inference_mode()
    def generate(
        self,
        chats: list[dict],
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
    ):
        ## TODO: 需要一个check函数，检查输入的history格式是否合法
        ## 比如，对于ASR任务，一定是: text-instruction/audio-instruction + audio-content, 我理解content和instruction是不能换位置的
        ## assistant前必须有user等等，我觉得最好做一下check

        assert output_type in ["text", "both"]

        history = self.prompt_manager.get_prompt(chats, output_type=output_type)
        
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_features = history.continuous_feature
        # print("text_input_ids =>", type(text_input_ids), text_input_ids.size())
        # print("audio_features =>", type(audio_features), audio_features[0].size())
        # text_input_ids => <class 'torch.Tensor'> torch.Size([1, 1425])
        # audio_features => <class 'list'> torch.Size([1, 750, 5120])
        generated_wav_tokens = []
        generated_text_tokens = []

        if output_type == "both":
            max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
        else:
            if max_new_tokens == -1:
                # max_new_tokens = 7500 - audio_input_ids.shape[1]
                max_new_tokens = 256

        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]
        # audio_features = [f.to(torch.cuda.current_device()).to(torch.float32) for f in audio_features] # just for infer by myself

        generated_wav_tokens, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            is_continuous_mask=is_continuous_mask,
            continous_feature=audio_features,
            output_type=output_type,
        )

        generated_wav_tokens = [
            t for t in generated_wav_tokens if t >= self.kimia_token_offset
        ]  #  filter out the illegal tokens

        generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
        generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset
        
        generated_text_tokens = [
            t for t in generated_text_tokens if t < self.kimia_token_offset
        ]
        generated_text = self.detokenize_text(generated_text_tokens)
        if self.detokenizer is not None and output_type == "both":
            generated_wav = self.detokenize_audio(generated_wav_tokens)
        else:
            generated_wav = None

        return generated_wav, generated_text
    
    # HQ grpo: 新增一个 直接支持token化输入的generate方法
    @torch.inference_mode()
    def generate_with_tokens(
        self,
        input_ids,
        text_input_ids,
        whisper_input_feature,
        is_continuous_mask,
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        # text_top_p=0.95,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
    ):
        ## TODO: 需要一个check函数，检查输入的history格式是否合法
        ## 比如，对于ASR任务，一定是: text-instruction/audio-instruction + audio-content, 我理解content和instruction是不能换位置的
        ## assistant前必须有user等等，我觉得最好做一下check

        # assert output_type in ["text", "both"]
        assert output_type in ["text"]

        # audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        # audio_features = history.continuous_feature

        # HQ grpo: 参数对应关系
        # input_ids # audio_input_ids
        # text_input_ids # text_input_ids
        # whisper_input_feature # audio_features
        # is_continuous_mask # is_continuous_mask

        generated_wav_tokens = []
        generated_text_tokens = []

        if output_type == "both":
            max_new_tokens = int(12.5 * 120) - input_ids.shape[1]
        else:
            if max_new_tokens == -1:
                # max_new_tokens = 7500 - input_ids.shape[1]
                max_new_tokens = 256

        audio_input_ids = input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        # whisper_input_feature = torch.tensor(whisper_input_feature) # HQ grpo
        # audio_features = [f.to(torch.cuda.current_device()) for f in whisper_input_feature]
        audio_features = [f for f in whisper_input_feature]
        # audio_features = [torch.tensor(f).to(torch.cuda.current_device()) for f in whisper_input_feature] # HQ grpo
        # local_rank = dist.get_rank()
        # print("1***************>len(audio_features) = ", len(audio_features), local_rank)

        generated_wav_tokens, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            # text_top_p=text_top_p,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            is_continuous_mask=is_continuous_mask,
            continous_feature=audio_features,
            output_type=output_type,
        )

        generated_wav_tokens = [
            t for t in generated_wav_tokens if t >= self.kimia_token_offset
        ]  #  filter out the illegal tokens

        generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
        generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset
        # print("generated_wav_tokens =>", generated_wav_tokens.size(), generated_wav_tokens)
        generated_text_tokens = [
            t for t in generated_text_tokens if t < self.kimia_token_offset
        ]
        # print("generated_text_tokens =>", generated_text_tokens.size(),generated_text_tokens)
        generated_text = self.detokenize_text(generated_text_tokens)
        
        if self.detokenizer is not None and output_type == "both":
            generated_wav = self.detokenize_audio(generated_wav_tokens)
        else:
            generated_wav = None

        return generated_wav, generated_text, generated_wav_tokens, generated_text_tokens

    def detokenize_audio(self, audio_tokens):
        if self.detokenizer is None:
            raise ValueError("Detokenizer is not initialized")
        self.detokenizer.clear_states()
        chunk_size = 30  # hard-coded right now
        first_chunk_size = 30
        cache_speech_collection = []
        audio_tokens = audio_tokens.to(torch.cuda.current_device())
        audio_tokens = audio_tokens.long()
        num_audio_tokens = audio_tokens.size(1)
        first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
        gen_speech = self.detokenizer.detokenize_streaming(
            first_chunk_semantic_tokens,
            is_final=(num_audio_tokens <= first_chunk_size),
            upsample_factor=4,
        )
        cache_speech_collection.append(gen_speech)

        if num_audio_tokens > first_chunk_size:
            res_semantic_tokens = audio_tokens[:, first_chunk_size:]
            for i in range(0, res_semantic_tokens.size(1), chunk_size):
                chunk_semantic_tokens = res_semantic_tokens[:, i : i + chunk_size]
                gen_speech = self.detokenizer.detokenize_streaming(
                    chunk_semantic_tokens,
                    upsample_factor=4,
                    is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
                )
                cache_speech_collection.append(gen_speech)

        gen_speech = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech

    def detokenize_text(self, text_tokens):
        valid_text_ids = []
        for x in text_tokens:
            if x == self.extra_tokens.kimia_text_eos:
                break
            valid_text_ids.append(x)
        return self.prompt_manager.text_tokenizer.decode(valid_text_ids)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--action", type=str, choices=["init_from_pretrained", "export_model"], default="init_from_pretrained")
    parser.add_argument("--output_dir", type=str, default="output/pretrained_hf")
    parser.add_argument("--input_dir", type=str, default="output/finetuned_hf")
    args = parser.parse_args()

    if args.action == "init_from_pretrained":

        model = KimiAudioModel.init_from_pretrained(args.model_name, model_load_kwargs={})

        os.makedirs(args.output_dir, exist_ok=True)
        # save model
        model.save_pretrained(args.output_dir)
    elif args.action == "export_model":
        KimiAudioModel.export_model(args.input_dir, args.output_dir)
    else:
        raise ValueError(f"Invalid action: {args.action}")