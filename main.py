import httpx
import json
import base64
import os
import glob
import shutil
import time
import argparse

from schema import AnnotationSegment
from tools.audio_clip import create_audio_clips
from tools.file_checker import check_file_exists
from tools.self_verificate import self_verificate

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

    parser = argparse.ArgumentParser(description="Multimodal LLM Annotation Pipeline")
    parser.add_argument("--include-past-anno", action="store_true", help="Include previously generated LLM annotation results in the Conversational Context JSON")
    args = parser.parse_args() # args.include_past_anno = True/False

    AUDIO_DIR_PATH = "/home/kanta/llm_annotation_app/data/audio"
    JSON_INPUT_DIR_PATH = "/home/kanta/llm_annotation_app/data/json_input"
    JSON_OUTPUT_BASE_DIR_PATH = "/home/kanta/llm_annotation_app/data/json_output"
    PROMPT_PATH = "/home/kanta/llm_annotation_app/Prompt/annotation_prompt_v1_jp.txt"
    HYPER_PARAMS_PATH = "/home/kanta/llm_annotation_app/hyperparams.json"

    json_output_dir_path = os.path.join(
        JSON_OUTPUT_BASE_DIR_PATH,
        (
            "dynamic_prompt_with_anno"
            if args.include_past_anno
            else "dynamic_prompt_without_anno"
        ),
    )
    os.makedirs(json_output_dir_path, exist_ok=True)

    audio_file_paths = sorted(glob.glob(os.path.join(AUDIO_DIR_PATH, "*.wav")))
    json_file_paths = sorted(glob.glob(os.path.join(JSON_INPUT_DIR_PATH, "*.json")))

    for json_file_path in json_file_paths:
        
        file_name = os.path.basename(json_file_path)
        file_stem = os.path.splitext(file_name)[0]

        audio_file_path = os.path.join(AUDIO_DIR_PATH, f"{file_stem}.wav")

        annotated_json_name = "annotation_" + file_name
        json_output_file_path = os.path.join(json_output_dir_path, annotated_json_name)

        if check_file_exists(json_output_file_path):
            continue
        
        start_time = time.time() # count the processing duration per scenario

        with open(json_file_path, "r", encoding="utf-8") as f:
            input_transcript = json.load(f)['utterances']

        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            user_prompt = f.read()

        with open(HYPER_PARAMS_PATH, "r", encoding="utf-8") as f:
            hparams = json.load(f)

        system_prompt = "You are an expert annotator for emotion and interaction analysis in dyadic conversation."

        clip_temp_dir_path, clip_audio_paths = create_audio_clips(audio_file_path, json_file_path)

        combined_annotations = []

        for target_utterance_idx, clip_audio_path in enumerate(clip_audio_paths):

            if args.include_past_anno:
                context_json = input_transcript[:target_utterance_idx] if target_utterance_idx==0 else combined_annotations
            else:
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

            # print(dynamic_user_prompt)

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

            print(f"\nSending a request for '{file_name}[{target_utterance_idx}]' to the local LLM server...\n")

            # --- API request to the local llama.cpp server ---
            with httpx.Client() as client:
                response = client.post(
                    "http://127.0.0.1:8080/v1/chat/completions",
                    json=payload,
                    timeout=None
                )

            output_json = json.loads(response.json()["choices"][0]["message"]["content"])
            combined_annotations.append(output_json)
        
        # --- combine jsons ---
        full_output_json = {
            "annotations": combined_annotations
        }

        # --- save json ---
        json_output_dir_path = os.path.join(JSON_OUTPUT_BASE_DIR_PATH, "dynamic_prompt_with_anno" if args.include_past_anno else "dynamic_prompt_without_anno")
        os.makedirs(json_output_dir_path, exist_ok=True)
        
        json_output_file_path = os.path.join(json_output_dir_path, annotated_json_name)

        with open(json_output_file_path, 'w', encoding='utf-8') as f:
            json.dump(full_output_json, f, indent=4, ensure_ascii=False)

        # --- verify generated json ---
        error_output_file_path = os.path.join(json_output_dir_path, "error", f"error_{annotated_json_name}")
        self_verificate(input_path=json_output_file_path, output_path=error_output_file_path)

        if os.path.exists(clip_temp_dir_path):
            shutil.rmtree(clip_temp_dir_path)

        # --- time count ---
        elapsed = time.time() - start_time
        print(f"----- {file_name}: {elapsed:.2f}s -----")


if __name__ == "__main__":
    main()