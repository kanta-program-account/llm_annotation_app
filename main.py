import httpx
import json
import base64
import os
import glob

from schema import FinalAnnotationResponse

def main():
    AUDIO_DIR_PATH = "/home/kanta/llm_annotation_app/data/audio"
    JSON_INPUT_DIR_PATH = "/home/kanta/llm_annotation_app/data/json_input"
    JSON_OUTPUT_DIR_PATH = "/home/kanta/llm_annotation_app/data/json_output"
    PROMPT_PATH = "/home/kanta/llm_annotation_app/Prompt/annotation_prompt_v1_jp.txt"
    HYPER_PARAMS_PATH = "/home/kanta/llm_annotation_app/hyperparams.json"

    audio_file_paths = sorted(glob.glob(os.path.join(AUDIO_DIR_PATH, "*.wav")))
    json_file_paths = sorted(glob.glob(os.path.join(JSON_INPUT_DIR_PATH, "*.json")))

    for audio_file_path, json_file_path in zip(audio_file_paths, json_file_paths):

        with open(audio_file_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        with open(json_file_path, "r", encoding="utf-8") as f:
            input_transcript_str = f.read()

        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            user_prompt = f.read()

        with open(HYPER_PARAMS_PATH, "r", encoding="utf-8") as f:
            hparams = json.load(f)

        system_prompt = "You are an expert annotator for emotion and interaction analysis in dyadic conversation."

        print(f"\nSending a request for {os.path.basename(json_file_path)} to the local LLM server...\n")

        payload = {
            "model": "Qwen3-Omni-30B-A3B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "text", "text": input_transcript_str},
                    {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}}
                ]
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "EmotionAnnotationSchema",
                    "strict": True,
                    "schema": FinalAnnotationResponse.model_json_schema()
                }
            },
            "cache_prompt": True,
        }

        payload.update(hparams) # hyperparams.json

        with httpx.Client() as client:
            response = client.post(
                "http://127.0.0.1:8080/v1/chat/completions",
                json=payload,
                timeout=None
            )

        # --- primary emotion, All emotion の包含ルールベース Verification ---
        #
        #
        # -------------------------------------------------------------------

        # --- save json ---
        output_json = json.loads(response.json()["choices"][0]["message"]["content"])

        annotated_json_name = "annotation_" + os.path.basename(json_file_path)

        output_file_path = os.path.join(JSON_OUTPUT_DIR_PATH, annotated_json_name)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4, ensure_ascii=False)
        # -----------------

if __name__ == "__main__":
    main()