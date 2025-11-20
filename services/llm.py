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
MODEL = os.getenv("OPENAI_MODEL")

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


def generate_channel_title(first_message: str, language: str = "ko") -> str:
    """
    사용자의 첫 메시지를 기반으로 AI 어시스턴트 채널 제목 생성
    
    Args:
        first_message: 사용자가 보낸 첫 메시지
        language: 제목 언어 (ko 또는 en)
    
    Returns:
        생성된 채널 제목 (최대 50자)
    """
    system_prompt = {
        "ko": "당신은 채팅 주제를 간결하게 요약하는 전문가입니다. 사용자의 질문을 보고 3-5 단어로 핵심 주제를 추출하세요.",
        "en": "You are an expert at summarizing chat topics concisely. Extract the key topic in 3-5 words from the user's question."
    }

    user_prompt = {
        "ko": f"다음 질문의 주제를 3-5 단어로 요약해주세요. 제목만 출력하세요:\n\n{first_message}",
        "en": f"Summarize the topic of this question in 3-5 words. Output only the title:\n\n{first_message}"
    }

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt.get(language, system_prompt["ko"])},
            {"role": "user", "content": user_prompt.get(language, user_prompt["ko"])},
        ],
        temperature=0.7,
        max_tokens=50
    )

    title = resp.choices[0].message.content.strip()

    # 따옴표 제거
    title = title.strip('"').strip("'")

    # 최대 50자 제한
    if len(title) > 50:
        title = title[:47] + "..."

    return title