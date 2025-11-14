from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.llm import refine
from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="Prizm LLM Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateReq(BaseModel):
    text: str
    target_lang: str = "en"

@app.post("/v1/translate")
def translate(req: TranslateReq):
    original = req.text
    # [MVP] LLM 단독: 후보 = 원문 그대로/간단 템플릿 (MT가 없으므로)
    candidate = f"(raw) {original}"
    refined = refine(original, candidate, req.target_lang)
    return {"result": refined}
