"""
Qdrant Vector Store Service
워크스페이스별로 벡터를 저장하고 관리합니다.
"""
import logging
import os
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self):
        # Qdrant API 키가 있으면 인증, 없으면 인증 없이
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_api_key:
            logger.info("Connecting to Qdrant with API key authentication")
            self.client = QdrantClient(
                host=os.getenv("QDRANT_HOST"),
                port=int(os.getenv("QDRANT_PORT")),
                api_key=qdrant_api_key
            )
        else:
            logger.info("Connecting to Qdrant without authentication")
            self.client = QdrantClient(
                host=os.getenv("QDRANT_HOST"),
                port=int(os.getenv("QDRANT_PORT")),
            )

        self.collection_name = "prizm_rag"
        self.vector_size = 1536  # OpenAI text-embedding-3-small

        # Collection 생성 (없으면)
        self._ensure_collection()

    def _ensure_collection(self):
        """Collection이 없으면 생성"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    def insert_vectors(
            self,
            workspace_id: int,
            file_id: int,
            chunks: List[Dict[str, Any]]
    ) -> int:
        """
        벡터를 Qdrant에 삽입
        
        Args:
            workspace_id: 워크스페이스 ID
            file_id: 파일 ID
            chunks: [{"text": str, "embedding": List[float], "chunk_index": int, "metadata": dict}]
        
        Returns:
            삽입된 벡터 개수
        """
        points = []

        for chunk in chunks:
            point_id = f"{workspace_id}_{file_id}_{chunk['chunk_index']}"

            payload = {
                "workspace_id": workspace_id,
                "file_id": file_id,
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {})
            }

            point = PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload=payload
            )
            points.append(point)

        # Batch insert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Inserted {len(points)} vectors for workspace={workspace_id}, file={file_id}")
        return len(points)

    def delete_by_file_id(self, workspace_id: int, file_id: int) -> int:
        """
        특정 workspace_id + file_id의 모든 벡터 삭제
        
        Args:
            workspace_id: 워크스페이스 ID
            file_id: 파일 ID
        
        Returns:
            삭제된 벡터 개수
        """
        # 삭제 전 개수 확인
        count_filter = Filter(
            must=[
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=workspace_id)
                ),
                FieldCondition(
                    key="file_id",
                    match=MatchValue(value=file_id)
                )
            ]
        )

        # 삭제
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=count_filter
        )

        logger.info(f"Deleted vectors for workspace={workspace_id}, file={file_id}")
        return 1  # Qdrant doesn't return count, so return 1 as success indicator

    def search(
            self,
            workspace_id: int,
            query_vector: List[float],
            limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        특정 워크스페이스 내에서 유사한 벡터 검색
        
        Args:
            workspace_id: 워크스페이스 ID
            query_vector: 쿼리 임베딩 벡터
            limit: 반환할 결과 개수
        
        Returns:
            검색 결과 리스트
        """
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=workspace_id)
                )
            ]
        )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "file_id": result.payload.get("file_id"),
                "chunk_index": result.payload.get("chunk_index"),
                "metadata": result.payload.get("metadata", {})
            }
            for result in results
        ]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Collection 통계 조회"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}