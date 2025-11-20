"""
AI Assistant Query Service
워크스페이스별 RAG 검색 및 응답 생성
"""
import logging
import os
from typing import List, Dict, Any

import openai

from .qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class AIAssistantService:
    def __init__(self):
        self.qdrant_service = QdrantService()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL")

    def create_query_embedding(self, query: str) -> List[float]:
        """
        쿼리 텍스트를 임베딩 벡터로 변환
        """
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=[query]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to create query embedding: {e}")
            raise

    def search_knowledge_base(
            self,
            workspace_id: int,
            query: str,
            limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        워크스페이스의 지식 베이스에서 관련 정보 검색
        
        Args:
            workspace_id: 워크스페이스 ID
            query: 사용자 질문
            limit: 검색 결과 개수
        
        Returns:
            검색 결과 리스트 (관련도 높은 순)
        """
        try:
            # 1. 쿼리 임베딩 생성
            query_vector = self.create_query_embedding(query)

            # 2. Qdrant에서 검색
            results = self.qdrant_service.search(
                workspace_id=workspace_id,
                query_vector=query_vector,
                limit=limit
            )

            logger.info(f"Found {len(results)} relevant chunks for workspace {workspace_id}")
            return results

        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []

    def generate_response(
            self,
            query: str,
            context_chunks: List[Dict[str, Any]],
            language: str = "ko"
    ) -> str:
        """
        RAG 기반 응답 생성
        
        Args:
            query: 사용자 질문
            context_chunks: 검색된 컨텍스트 청크
            language: 응답 언어 (ko, en 등)
        
        Returns:
            AI 생성 응답
        """
        try:
            # 1. 컨텍스트 구성
            if context_chunks:
                context_text = "\n\n".join([
                    f"[문서 {i + 1}] (관련도: {chunk['score']:.2f})\n{chunk['text']}"
                    for i, chunk in enumerate(context_chunks)
                ])
            else:
                context_text = "관련 문서를 찾을 수 없습니다."

            # 2. 시스템 프롬프트
            system_prompt = self._get_system_prompt(language)

            # 3. ChatGPT API 호출
            response = openai.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
다음 문서들을 참고하여 질문에 답변해주세요:

{context_text}

질문: {query}
"""}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            logger.info(f"Generated response (length: {len(answer)})")

            return answer

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._get_error_message(language)

    def chat(
            self,
            workspace_id: int,
            query: str,
            language: str = "ko",
            search_limit: int = 5
    ) -> Dict[str, Any]:
        """
        AI 어시스턴트 전체 처리 플로우
        
        Args:
            workspace_id: 워크스페이스 ID
            query: 사용자 질문
            language: 응답 언어
            search_limit: 검색 결과 개수
        
        Returns:
            {
                "answer": "AI 응답",
                "sources": [검색된 소스],
                "has_context": bool
            }
        """
        try:
            # 1. 지식 베이스 검색
            context_chunks = self.search_knowledge_base(
                workspace_id=workspace_id,
                query=query,
                limit=search_limit
            )

            # 2. 응답 생성
            answer = self.generate_response(
                query=query,
                context_chunks=context_chunks,
                language=language
            )

            # 3. 소스 정보 정리
            sources = [
                {
                    "file_id": chunk["file_id"],
                    "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "score": chunk["score"]
                }
                for chunk in context_chunks[:3]  # 상위 3개만
            ]

            return {
                "answer": answer,
                "sources": sources,
                "has_context": len(context_chunks) > 0
            }

        except Exception as e:
            logger.error(f"AI chat failed: {e}")
            return {
                "answer": self._get_error_message(language),
                "sources": [],
                "has_context": False
            }

    def _get_system_prompt(self, language: str) -> str:
        """시스템 프롬프트 생성"""
        prompts = {
            "ko": """당신은 워크스페이스의 문서를 학습한 AI 어시스턴트입니다.

역할:
- 제공된 문서 내용을 기반으로 정확하고 도움이 되는 답변을 제공합니다
- 문서에 없는 내용은 추측하지 말고 "문서에서 관련 정보를 찾을 수 없습니다"라고 답변합니다
- 답변은 친절하고 전문적인 톤으로 작성합니다
- 가능하면 문서의 구체적인 내용을 인용하여 답변합니다

답변 형식:
- 명확하고 구조화된 답변
- 필요시 번호나 불릿 포인트 사용
- 한국어로 답변""",

            "en": """You are an AI assistant trained on workspace documents.

Role:
- Provide accurate and helpful answers based on the provided documents
- If information is not in the documents, say "I couldn't find relevant information in the documents"
- Use a friendly and professional tone
- Cite specific document content when possible

Answer format:
- Clear and structured responses
- Use numbered lists or bullet points when needed
- Answer in English"""
        }

        return prompts.get(language, prompts["ko"])

    def _get_error_message(self, language: str) -> str:
        """에러 메시지 생성"""
        messages = {
            "ko": "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "en": "Sorry, a temporary error occurred. Please try again later."
        }
        return messages.get(language, messages["ko"])