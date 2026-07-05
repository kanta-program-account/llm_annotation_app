import httpx
import json
import base64

from schema import FinalAnnotationResponse

def main():
    with open("/home/kanta/llm_annotation_app/data/audio/scn-0185_pair-0004.wav", "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    with open("/home/kanta/llm_annotation_app/data/json_input/scn_0185_pair_0004.json", "r", encoding="utf-8") as f:
        input_transcript_str = f.read()

    with open("/home/kanta/llm_annotation_app/Prompt/annotation_prompt_v1_jp.txt", "r", encoding="utf-8") as f:
        user_prompt = f.read()

    system_prompt = "You are an expert annotator for emotion and interaction analysis in dyadic conversation."

    print("\nSending a request to the local LLM server...\n")

    response = httpx.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={
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
            "temperature": 0.7,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "EmotionAnnotationSchema",
                    "strict": True,
                    "schema": FinalAnnotationResponse.model_json_schema()
                }
            },
        },
        timeout=1800 # 30min
    )

    print(response.json()["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
