# python
import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드 (루트에 있으면 load_dotenv() 만으로도 충분)
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되어 있지 않습니다. 환경변수로 설정하거나 프로젝트 루트에 `.env` 파일을 만들고 "
        "OPENAI_API_KEY=your_key 를 추가하세요."
    )

client = OpenAI(api_key=API_KEY)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM = (
    "You are a bilingual software engineer translator for a developer chat. "
    "Keep code, library, framework and API names untranslated. "
    "Make the text concise and natural for chat."
)

def refine(original: str, candidate: str, target_lang: str) -> str:
    prompt = (
        f"Original Text:\n{original}\n\n"
        f"Target language: {target_lang}\n"
        "Translate the 'Original Text' into the 'Target language'. "
        "Then, refine the translated text to be natural for a developer chat, preserving technical terms. "
        "Your final output should only be the refined translation, without any additional commentary or labels."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()
