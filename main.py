import httpx
import json
import base64
import os
import glob
import shutil
import time

from schema import AnnotationSegment
from tools.audio_clip import create_audio_clips

def main():
    """Run the multimodal LLM emotion and interaction annotation pipeline.

    Processes pairs of audio recordings (.wav) and conversational transcript JSONs
    by splitting the audio into utterance-level clips, passing each clip along
    with its historical text context to a local Qwen3-Omni model, and merging the
    resulting structured emotion annotations into a consolidated output JSON file.

    Args:
        None

    Returns:
        None
    """
    AUDIO_DIR_PATH = "/home/kanta/llm_annotation_app/data/audio"
    JSON_INPUT_DIR_PATH = "/home/kanta/llm_annotation_app/data/json_input"
    JSON_OUTPUT_DIR_PATH = "/home/kanta/llm_annotation_app/data/json_output"
    PROMPT_PATH = "/home/kanta/llm_annotation_app/Prompt/annotation_prompt_v1_jp.txt"
    HYPER_PARAMS_PATH = "/home/kanta/llm_annotation_app/hyperparams.json"

    audio_file_paths = sorted(glob.glob(os.path.join(AUDIO_DIR_PATH, "*.wav")))
    json_file_paths = sorted(glob.glob(os.path.join(JSON_INPUT_DIR_PATH, "*.json")))

    for audio_file_path, json_file_path in zip(audio_file_paths, json_file_paths):
        
        file_name = os.path.basename(json_file_path)
        
        start_time = time.time() # count the processing duration per scenario

        with open(json_file_path, "r", encoding="utf-8") as f:
            input_transcript = json.load(f)
            input_transcript_str = json.dumps(input_transcript, ensure_ascii=False)

        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            user_prompt = f.read()

        with open(HYPER_PARAMS_PATH, "r", encoding="utf-8") as f:
            hparams = json.load(f)

        system_prompt = "You are an expert annotator for emotion and interaction analysis in dyadic conversation."

        clip_temp_dir_path, clip_audio_paths = create_audio_clips(audio_file_path, json_file_path)

        combined_annotations = []

        for target_utterance_idx, clip_audio_path in enumerate(clip_audio_paths):
            context_json = input_transcript[:target_utterance_idx]
            anno_target_json = input_transcript[target_utterance_idx]

            context_str = json.dumps(context_json, ensure_ascii=False, indent=2)
            anno_target_str = json.dumps(anno_target_json, ensure_ascii=False, indent=2)

            dynamic_user_prompt = (
                f"{user_prompt}\n\n"
                f"### Conversational Context JSON\n"
                f"*NOTE: The following past data is for contextual understanding only. Do NOT annotate these segments.*\n"
                f"{context_str}\n\n"
                f"### Target Utterance JSON\n"
                f"*NOTE: Only this specific utterance and its accompanying audio interval must be the subject of your annotation.*\n"
                f"{anno_target_str}"
            )

            with open(clip_audio_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "model": "Qwen3-Omni-30B-A3B-Instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", 
                    "content": [
                        {"type": "text", "text": dynamic_user_prompt},
                        {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}}
                    ]
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "EmotionAnnotationSchema",
                        "strict": True,
                        "schema": AnnotationSegment.model_json_schema()
                    }
                },
                "cache_prompt": True,
            }

            payload.update(hparams)

            print("\nSending a request to the local LLM server...\n")

            with httpx.Client() as client:
                response = client.post(
                    "http://127.0.0.1:8080/v1/chat/completions",
                    json=payload,
                    timeout=None
                )

            output_json = json.loads(response.json()["choices"][0]["message"]["content"])
            combined_annotations.append(output_json)

        full_output_json = {
            "annotations": combined_annotations
        }

        annotated_json_name = "annotation_" + file_name

        output_file_path = os.path.join(JSON_OUTPUT_DIR_PATH, annotated_json_name)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(full_output_json, f, indent=4, ensure_ascii=False)

        if os.path.exists(clip_temp_dir_path):
            shutil.rmtree(clip_temp_dir_path)

        # --- time count ---
        elapsed = time.time() - start_time
        print(f"----- {file_name}: {elapsed:.2f}s -----")

if __name__ == "__main__":
    main()