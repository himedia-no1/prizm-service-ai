"""
RAG Document Processing Service
파일 다운로드 → 파싱 → Chunking → Embedding → Qdrant 저장
"""
import logging
import os
import tempfile
from typing import List, Dict, Any
from urllib.parse import urlparse

import boto3
import openai
from botocore.client import Config
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_analyzer import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_xlsx,
    extract_text_from_pptx
)
from .qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class RAGProcessor:
    def __init__(self):
        self.qdrant_service = QdrantService()

        # OpenAI 설정
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # S3/MinIO 설정 (Spring과 동일한 방식)
        s3_endpoint = os.getenv("S3_ENDPOINT")
        s3_region = os.getenv("S3_REGION")

        s3_config = {
            'service_name': 's3',
            'aws_access_key_id': os.getenv("S3_ACCESS_KEY"),
            'aws_secret_access_key': os.getenv("S3_SECRET_KEY"),
            'region_name': s3_region,
            'config': Config(signature_version='s3v4')
        }

        # MinIO인 경우 (endpoint 있으면)
        if s3_endpoint:
            s3_config['endpoint_url'] = s3_endpoint
            # MinIO는 path-style access 필요
            s3_config['config'] = Config(
                signature_version='s3v4',
                s3={'addressing_style': 'path'}
            )

        self.s3_client = boto3.client(**s3_config)
        self.bucket_name = os.getenv("S3_BUCKET")

        # Text Splitter 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def download_file_from_minio(self, file_key: str) -> str:
        """
        S3/MinIO에서 파일 다운로드
        
        Args:
            file_key: S3/MinIO 파일 키 (예: files/uuid.pdf)
        
        Returns:
            로컬 임시 파일 경로
        """
        try:
            # 임시 파일 생성
            suffix = os.path.splitext(file_key)[1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_path = temp_file.name
            temp_file.close()

            # S3/MinIO에서 다운로드
            logger.info(f"Downloading file from S3/MinIO: {file_key}")
            self.s3_client.download_file(self.bucket_name, file_key, temp_path)

            logger.info(f"File downloaded successfully to {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"Failed to download file from S3/MinIO: {e}")
            raise

    def download_file_from_url(self, presigned_url: str) -> str:
        """
        Presigned URL에서 파일 다운로드 (대안)
        
        Args:
            presigned_url: S3 Presigned URL
        
        Returns:
            로컬 임시 파일 경로
        """
        import requests

        try:
            # URL에서 확장자 추출
            parsed = urlparse(presigned_url)
            path = parsed.path
            suffix = os.path.splitext(path)[1]

            # 임시 파일 생성
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_path = temp_file.name

            # 다운로드
            logger.info(f"Downloading file from URL: {presigned_url[:50]}...")
            response = requests.get(presigned_url, stream=True)
            response.raise_for_status()

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"File downloaded successfully to {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"Failed to download file from URL: {e}")
            raise

    def extract_text_from_file(self, file_path: str, file_name: str) -> str:
        """
        파일에서 텍스트 추출
        
        Args:
            file_path: 로컬 파일 경로
            file_name: 원본 파일명 (확장자 판단용)
        
        Returns:
            추출된 텍스트
        """
        extension = os.path.splitext(file_name)[1].lower()

        try:
            if extension == '.pdf':
                return extract_text_from_pdf(file_path)
            elif extension == '.docx':
                return extract_text_from_docx(file_path)
            elif extension == '.xlsx':
                return extract_text_from_xlsx(file_path)
            elif extension == '.pptx':
                return extract_text_from_pptx(file_path)
            elif extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise ValueError(f"Unsupported file type: {extension}")
        except Exception as e:
            logger.error(f"Failed to extract text from {file_name}: {e}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 전체 텍스트
        
        Returns:
            청크 리스트
        """
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        OpenAI Embedding 생성
        
        Args:
            texts: 텍스트 리스트
        
        Returns:
            임베딩 벡터 리스트
        """
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(f"Created {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise

    def process_file_for_rag(
            self,
            workspace_id: int,
            file_id: int,
            file_key: str,
            file_name: str
    ) -> Dict[str, Any]:
        """
        RAG 처리 전체 파이프라인
        
        Args:
            workspace_id: 워크스페이스 ID
            file_id: 파일 ID
            file_key: MinIO 파일 키
            file_name: 원본 파일명
        
        Returns:
            처리 결과 통계
        """
        temp_file_path = None

        try:
            # 1. 파일 다운로드
            logger.info(f"Step 1: Downloading file - workspace={workspace_id}, file={file_id}")
            temp_file_path = self.download_file_from_minio(file_key)

            # 2. 텍스트 추출
            logger.info(f"Step 2: Extracting text from {file_name}")
            text = self.extract_text_from_file(temp_file_path, file_name)

            if not text or len(text.strip()) < 10:
                raise ValueError("Extracted text is too short or empty")

            # 3. Chunking
            logger.info(f"Step 3: Chunking text (length={len(text)})")
            chunks = self.chunk_text(text)

            if not chunks:
                raise ValueError("No chunks created from text")

            # 4. Embedding 생성
            logger.info(f"Step 4: Creating embeddings for {len(chunks)} chunks")
            embeddings = self.create_embeddings(chunks)

            # 5. Qdrant에 저장
            logger.info(f"Step 5: Inserting vectors into Qdrant")
            chunks_data = [
                {
                    "text": chunk,
                    "embedding": embedding,
                    "chunk_index": idx,
                    "metadata": {
                        "file_name": file_name,
                        "chunk_length": len(chunk)
                    }
                }
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]

            vector_count = self.qdrant_service.insert_vectors(
                workspace_id=workspace_id,
                file_id=file_id,
                chunks=chunks_data
            )

            logger.info(f"RAG processing completed successfully: {vector_count} vectors inserted")

            return {
                "success": True,
                "workspace_id": workspace_id,
                "file_id": file_id,
                "chunks_count": len(chunks),
                "vectors_count": vector_count,
                "text_length": len(text)
            }

        except Exception as e:
            logger.error(f"RAG processing failed: {e}")
            raise

        finally:
            # 임시 파일 삭제
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Temporary file deleted: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")

    def delete_vectors_by_file(self, workspace_id: int, file_id: int) -> Dict[str, Any]:
        """
        특정 파일의 모든 벡터 삭제
        
        Args:
            workspace_id: 워크스페이스 ID
            file_id: 파일 ID
        
        Returns:
            삭제 결과
        """
        try:
            count = self.qdrant_service.delete_by_file_id(workspace_id, file_id)
            logger.info(f"Deleted vectors for workspace={workspace_id}, file={file_id}")

            return {
                "success": True,
                "workspace_id": workspace_id,
                "file_id": file_id,
                "deleted_count": count
            }

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise