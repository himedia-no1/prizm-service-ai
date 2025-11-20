import logging

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from services.ai_assistant_service import AIAssistantService
from services.document_analyzer import analyze_document
from services.llm import refine, generate_channel_title
from services.rag_processor import RAGProcessor

load_dotenv()

# Logging 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="prizm-service-ai",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# RAG Processor 초기화
rag_processor = RAGProcessor()

# AI Assistant 초기화
ai_assistant = AIAssistantService()


class TranslateReq(BaseModel):
    text: str
    target_lang: str = "en"


class AnalyzeDocumentReq(BaseModel):
    fileUrl: str
    fileName: str
    summaryLanguage: str = "en"


class ProcessRAGRequest(BaseModel):
    workspace_id: int
    file_id: int
    file_key: str
    file_name: str
    callback_url: str  # Spring Boot 콜백 URL


class DeleteVectorsRequest(BaseModel):
    workspace_id: int
    file_id: int


class AIChatRequest(BaseModel):
    workspace_id: int
    query: str
    language: str = "ko"
    search_limit: int = 5


class GenerateTitleRequest(BaseModel):
    first_message: str
    language: str = "ko"


@app.post("/ai/translate")
def translate(req: TranslateReq):
    original = req.text
    # [MVP] LLM 단독: 후보 = 원문 그대로/간단 템플릿 (MT가 없으므로)
    candidate = f"(raw) {original}"
    refined = refine(original, candidate, req.target_lang)
    return {"result": refined}


@app.post("/ai/analyze")
def analyze_document_endpoint(req: AnalyzeDocumentReq):
    try:
        summary = analyze_document(
            req.fileUrl,
            req.fileName,
            req.summaryLanguage
        )
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_rag_background(req: ProcessRAGRequest):
    """
    RAG 처리 백그라운드 작업 + Spring Boot 콜백
    """
    try:
        logger.info(f"Starting RAG processing for file_id={req.file_id}")

        # RAG 처리
        result = rag_processor.process_file_for_rag(
            workspace_id=req.workspace_id,
            file_id=req.file_id,
            file_key=req.file_key,
            file_name=req.file_name
        )

        logger.info(f"RAG processing completed: {result}")

        # Spring Boot에 SUCCESS 콜백
        async with httpx.AsyncClient() as client:
            callback_data = {
                "file_id": req.file_id,
                "status": "SUCCESS",
                "chunks_count": result["chunks_count"],
                "vectors_count": result["vectors_count"]
            }

            response = await client.post(req.callback_url, json=callback_data, timeout=10.0)
            logger.info(f"Callback sent successfully: {response.status_code}")

    except Exception as e:
        logger.error(f"RAG processing failed: {e}")

        # Spring Boot에 FAILURE 콜백
        try:
            async with httpx.AsyncClient() as client:
                callback_data = {
                    "file_id": req.file_id,
                    "status": "FAILURE",
                    "error": str(e)
                }

                await client.post(req.callback_url, json=callback_data, timeout=10.0)
                logger.info("Failure callback sent")
        except Exception as callback_error:
            logger.error(f"Failed to send callback: {callback_error}")


@app.post("/ai/rag")
async def process_rag_file(req: ProcessRAGRequest, background_tasks: BackgroundTasks):
    """
    RAG 파일 처리 (비동기)
    
    Spring Boot에서 호출:
    - workspace_id: 워크스페이스 ID
    - file_id: 파일 ID
    - file_key: MinIO 파일 키 (예: files/uuid.pdf)
    - file_name: 원본 파일명
    - callback_url: 완료 시 콜백 URL
    """
    try:
        # 백그라운드에서 처리
        background_tasks.add_task(process_rag_background, req)

        return {
            "status": "processing",
            "file_id": req.file_id,
            "message": "RAG processing started in background"
        }

    except Exception as e:
        logger.error(f"Failed to start RAG processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ai/rag/workspaces/{workspace_id}/files/{file_id}")
def delete_rag_vectors(workspace_id: int, file_id: int):
    """
    특정 파일의 모든 벡터 삭제
    
    Args:
        workspace_id: 워크스페이스 ID
        file_id: 파일 ID
    """
    try:
        result = rag_processor.delete_vectors_by_file(
            workspace_id=workspace_id,
            file_id=file_id
        )

        return {
            "status": "success",
            "workspace_id": workspace_id,
            "file_id": file_id,
            "message": "Vectors deleted successfully"
        }

    except Exception as e:
        logger.error(f"Failed to delete vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/chat")
def ai_chat(req: AIChatRequest):
    """
    AI 어시스턴트 채팅
    
    워크스페이스의 학습된 문서를 기반으로 질문에 답변합니다.
    
    Args:
        req: AIChatRequest
            - workspace_id: 워크스페이스 ID
            - query: 사용자 질문
            - language: 응답 언어 (ko, en)
            - search_limit: 검색할 문서 청크 개수
    
    Returns:
        {
            "answer": "AI 응답",
            "sources": [소스 정보],
            "has_context": bool
        }
    """
    try:
        result = ai_assistant.chat(
            workspace_id=req.workspace_id,
            query=req.query,
            language=req.language,
            search_limit=req.search_limit
        )

        return result

    except Exception as e:
        logger.error(f"AI chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/chat/title")
def generate_title(req: GenerateTitleRequest):
    """
    AI 어시스턴트 채널 제목 생성
    
    사용자의 첫 메시지를 기반으로 채널 제목을 자동 생성합니다.
    
    Args:
        req: GenerateTitleRequest
            - first_message: 사용자의 첫 메시지
            - language: 제목 언어 (ko, en)
    
    Returns:
        {
            "title": "생성된 채널 제목"
        }
    """
    try:
        title = generate_channel_title(
            first_message=req.first_message,
            language=req.language
        )

        return {"title": title}

    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))