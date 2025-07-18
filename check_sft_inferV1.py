from kimia_infer.api.kimia import KimiAudio
import json
import time
import os

"""
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.model --model_name "moonshotai/Kimi-Audio-7B" \
--action "export_model" \
--input_dir "/root/autodl-tmp/dataset/output/checkpoint-200" \
--output_dir "/root/autodl-tmp/dataset/output/finetuned_hf_for_inference"
"""

# understanding ds(非semantic)
# JSON_PATH  = "/root/autodl-tmp/dataset/start/material/val60_km_sys.jsonl"
# JSON_PATH  = "/root/autodl-tmp/dataset/start/material/val_km.jsonl"
JSON_PATH  = "/root/autodl-tmp/dataset/start/material/val60_km_val_grponothink.jsonl"

# MODEL_PATH = "/root/autodl-tmp/dataset/output/finetuned_hf_for_inference"
MODEL_PATH = "/root/autodl-tmp/models/Kimi-Audio-7B-Instruct"
# MODEL_PATH = "/root/autodl-tmp/dataset/output/MAP1-After"
# OUT_PATH   = "/root/autodl-tmp/dataset/output/KimiM—test_age-60s-audios.jsonl"
# OUT_PATH   = "/root/autodl-tmp/dataset/output/KimiM—test_age-90s-audios.jsonl"
# OUT_PATH   = "/root/autodl-tmp/dataset/output/kimi_audio_7b_instruct_v2_val_sys_60s-infer1.jsonl"
OUT_PATH   = "/root/autodl-tmp/dataset/output/infer-test_val60_km_val_grponothink.jsonl"


print("loading model")
model = KimiAudio(model_path=MODEL_PATH, load_detokenizer=False)

print("sampling config")
sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

samples = []
with open(JSON_PATH, "r", encoding="utf-8") as f:
    for line in f.readlines():
        samples.append(json.loads(line))
print(f"📚 {len(samples)} samples loaded.")


# ==== 推理循环（支持断点续推） ====
correct = 0
start = time.time()

# ---- 检查已有推理结果 ----
existing_ids = set()
existing_id_reply = {} # HQ add

if os.path.exists(OUT_PATH):
    print(f"⚡️ Found existing result file {OUT_PATH}. Resuming...")
    with open(OUT_PATH, "r", encoding="utf-8") as f_exist:
        for line in f_exist:
            try:
                data = json.loads(line)
                existing_ids.add(data["mid"])
                existing_id_reply[data["mid"]] = data # HQ add
            except:
                continue
    print(f"⚡️ {len(existing_ids)} samples already completed.")
else:
    print(f"🆕 No existing result file. Start fresh.")

# ---- 追加模式打开文件 ----
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "a", encoding="utf-8") as fout:
    for idx, ex in enumerate(samples):
        messages = []
        if not ex["mid"] in existing_ids: # HQ modify
            messages = list(filter(lambda item: item["role"] != "assistant", ex["conversation"]))
            _, text = model.generate(messages, **sampling_params, output_type="text")
            reply = text
        else:
            reply = existing_id_reply[ex["mid"]]["reply"] # HQ add
            
        # --- 评估 ---
        pred = reply
        gt   = list(filter(lambda item: item["role"] == "assistant", ex["conversation"] ))[0]["content"]
        # correct += (pred == gt)
        correct += (pred == gt.replace("<answer>", "").replace("</answer>", "").strip())
        
        if not ex["mid"] in existing_ids: 
            fout.write(json.dumps({
                # "id": ex["id"],
                "origin_gt": gt,
                "gt": gt.replace("<answer>", "").replace("</answer>", "").strip(),
                "pred": pred,
                "conversation": json.dumps(messages, ensure_ascii=False),
                "reply": reply,
                "variant": "SFT",
                "mid": ex["mid"], # HQ add
            }, ensure_ascii=False) + "\n")
            fout.flush()

        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(samples)}] Acc {correct/(idx+1):.2%}")

# ==== 总结 ====
acc = correct / len(samples)
print(f"\n🎯 {'SFT'} accuracy {acc:.2%} | elapsed {(time.time()-start)/60:.1f} min")
print("Results saved to", OUT_PATH)















