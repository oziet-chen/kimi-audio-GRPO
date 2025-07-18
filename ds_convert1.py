
import os

import json
import re
from pathlib import Path
"""


JSON
id227
audio_id27101
lyric_id27101
singer_name江北
song_name贪杯(DJ阿卓版)
mid002cxzO83RtXHy
published_date2023-06-13
questionBased on the given audio, song name ：<此处插入歌曲名称>，lyrics ：<此处插入歌词文本>, singer's name:<此处插入歌手名称>, and published data：<此处插入发行日期>, which age group is most likely to prefer this song?
multi_choice
0Youth (mainly under 22)
1Young Adults (mainly 23-30)
2Middle-aged Adults (mainly 31-40)
3Seniors (mainly 41+)
answer1
datasetMAP
audio_typeslice
categoryReasoning
sub_categoryAge Group Preference



task_typeunderstanding
conversation
0
roleuser
message_typetext
contentPlease transcribe the spoken content into written text.
1
roleuser
message_typeaudio
contentfinetune_codes/demo_data/audio_understanding/audios/librispeech_1263-139804-0005.flac
2
roleassistant
message_typetext
contentearly in the morning certain crops are planted and are harvested at night two or more days are required for maturing other crops actually the people of brief raise their crops with less labor than is required amongst us



goon todo:
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.extract_semantic_codes --model_name_or_path "/root/autodl-tmp/models/Kimi-Audio-7B" --input_file "/root/autodl-tmp/dataset/start/material/val60_km_test_grponothink.jsonl" --output_file "/root/autodl-tmp/dataset/start/material/val60_km_test_grponothink_semantic_codes.jsonl"

"""

# SFT_NORMAL = True 
SFT_NORMAL = False 
GRPO_NO_THINK = True

if not SFT_NORMAL:
    if GRPO_NO_THINK:
        QUESTION_TEMPLATE = (
            "{Question}\n"
            # "Please think about this question as if you were a human pondering deeply. " # no think annotation
            # "Make sure to carefully consider both the visual and audio information before answering. "
            "Make sure to carefully consider the audio information before answering. " # HQ modify
            # "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions " # no think annotation
            # "It's encouraged to include self-reflection or verification in the reasoning process. " # no think annotation
            # "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags." # no think annotation
            "Just give your final answer between the <answer> </answer> tags." # think annotation
        )
    else:
        QUESTION_TEMPLATE = (
            "{Question}\n"
            "Please think about this question as if you were a human pondering deeply. " # no think annotation
            # "Make sure to carefully consider both the visual and audio information before answering. "
            "Make sure to carefully consider the audio information before answering. " # HQ modify
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions " # no think annotation
            "It's encouraged to include self-reflection or verification in the reasoning process. " # no think annotation
            "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags." # no think annotation
            # "Just give your final answer between the <answer> </answer> tags." # think annotation
        )

    TYPE_TEMPLATE = {
        # "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
        "multiple choice": " Please provide only the content of a single option within the <answer> </answer> tag (for example, consistent with one of the options).", # HQ modify
    }

    SOLUTION_TEMPLATE = ("<answer>{Solution}</answer>")
else:
    ## SFT_NORMAL
    # 这里参考Echoink BASE model的prompt强化模版 设置内容如下
    QUESTION_TEMPLATE = (
        "{Question}\n"
        # "Please answer the following question based on the given image and audio."
        "Please answer the following question based on the given audio." # HQ modify
    )

    TYPE_TEMPLATE = {
        # "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
        # "multiple choice": " Please provide only the content of a single option within the <answer> </answer> tag (for example, consistent with one of the options).", # HQ modify
        "multiple choice": " Please provide only the content of a single option (for example, consistent with one of the options).", # HQ modify
    }

    SOLUTION_TEMPLATE = ("{Solution}")

# SYSTEM_PROMPT_THINKING = "As a music market analysis expert, you need to comprehensively analyze target audiences by integrating audio characteristics, publish data, lyric content, song name, and the artist background. Follow these steps: 1. Analyze musical features including rhythm, melody, and arrangement style. 2. Interpret lyrical themes, emotional tendencies, and language complexity. 3. Consider the artist's typical listener profile and genre characteristics.4. Synthesize these elements to select the best-matching age group from provided options. 5. Final answer must be strictly enclosed in <answer> tags containing only the chosen option, without explanations."
SYSTEM_PROMPT_SFT = "As a music market analysis expert, you need to comprehensively analyze target audiences by integrating audio characteristics, publish data, lyric content, song name, and the artist background. Follow these steps: 1. Analyze musical features including rhythm, melody, and arrangement style. 2. Interpret lyrical themes, emotional tendencies, and language complexity. 3. Consider the artist's typical listener profile and genre characteristics.4. Synthesize these elements to select the best-matching age group from provided options. 5. The final answer only includes the selected option and does not contain explanations."

def map_ds_convert_to_understanding_ds(map_ds_path, understanding_ds_path, ds_dir, audios_subdir="audios", is_add_sys=False):
    new_data_items = []
    with open(map_ds_path,"r") as map_ds_file:
        for line in map_ds_file.readlines():
            data_item = json.loads(line)

            mid = data_item["mid"]
            _id = data_item["id"]
            song_name = data_item["song_name"]
            singer_name = data_item["singer_name"]
            published_date = data_item["published_date"]
            question = data_item["question"]
            multi_choice = data_item["multi_choice"]
            answer = data_item["answer"]

            aud_path = "{}/{}/{}.wav".format(ds_dir,audios_subdir,mid)  
            lyric_path = ("{}/lyrics/{}.txt".format(ds_dir, mid))
            lyrics = ""
            if Path(lyric_path).exists():
                with open(lyric_path, 'r') as file:
                    lyrics = file.read()
                    lyrics = re.sub(r'\[\d+:\d+.\d+\]', '', lyrics) # 过滤调时间戳信息
            else:
                raise
                pass
            
            text_prompt = question.replace("<此处插入歌曲名称>", song_name).replace("<此处插入歌词文本>", lyrics).replace("<此处插入歌手名称>", singer_name).replace("<此处插入发行日期>", published_date) + " Options:\n" + "\n".join(multi_choice)
            prompt = QUESTION_TEMPLATE.format(Question=text_prompt) + TYPE_TEMPLATE.get("multiple choice", "")
            solution = SOLUTION_TEMPLATE.format(Solution=multi_choice[int(answer)])

            conversations = []
            if is_add_sys:
                conversations.append({
                    "role": "user",
                    "message_type": "text",
                    "content":SYSTEM_PROMPT_SFT
                })
            
            conversations.append({
                "role": "user",
                "message_type": "text",
                "content":prompt
            })
            conversations.append({
                "role": "user",
                "message_type": "audio",
                "content":aud_path
            })

            if SFT_NORMAL:
                conversations.append({
                    "role": "assistant",
                    "message_type": "text",
                    "content":solution
                })

            new_data_item = {
                "task_type": "understanding", 
                "mid": mid,
                "id": _id,
                "problem_type": "multiple choice",
                "solution": solution,
                "conversation": conversations
            }
           

            print(mid)
            new_data_items.append(new_data_item)

    if Path(understanding_ds_path).exists():
        return understanding_ds_path
    
    with open(understanding_ds_path, "w+") as understanding_ds_file:
        for item in new_data_items:
            understanding_ds_file.write(f"{json.dumps(item)}\n")
    return understanding_ds_path


if __name__ == "__main__":
    GRPO_NO_THINK_STR = ""
    if GRPO_NO_THINK:
        GRPO_NO_THINK_STR = "nothink"
    
    # map_ds_convert_to_understanding_ds("/root/autodl-tmp/dataset/start/material/train.jsonl", "/root/autodl-tmp/dataset/start/material/train_km.jsonl", "/root/autodl-tmp/dataset/start")
    # map_ds_convert_to_understanding_ds("/root/autodl-tmp/dataset/start/material/val.jsonl", "/root/autodl-tmp/dataset/start/material/val_km.jsonl", "/root/autodl-tmp/dataset/start")
    # map_ds_convert_to_understanding_ds("/root/autodl-tmp/dataset/start/material/train_small.jsonl", "/root/autodl-tmp/dataset/start/material/train_small_km.jsonl", "/root/autodl-tmp/dataset/start")

    # map_ds_convert_to_understanding_ds("/root/autodl-tmp/dataset/start/material/train60.jsonl", "/root/autodl-tmp/dataset/start/material/train60_km.jsonl", "/root/autodl-tmp/dataset/start", "audios60")

    # map_ds_convert_to_understanding_ds("/root/autodl-tmp/dataset/start/material/val60.jsonl", "/root/autodl-tmp/dataset/start/material/val60_km.jsonl", "/root/autodl-tmp/dataset/start", "audios60")
    # map_ds_convert_to_understanding_ds("/root/autodl-tmp/dataset/start/material/train.jsonl", "/root/autodl-tmp/dataset/start/material/train60_km_test_grpo.jsonl", "/root/autodl-tmp/dataset/start", "audios60",is_add_sys=False)
    map_ds_convert_to_understanding_ds("/root/autodl-tmp/dataset/start/material/val.jsonl", "/root/autodl-tmp/dataset/start/material/val60_km_test_grpo{}.jsonl".format(GRPO_NO_THINK_STR), "/root/autodl-tmp/dataset/start", "audios60",is_add_sys=False)


