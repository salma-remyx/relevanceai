from __future__ import annotations

from .._client import RelevanceAI, AsyncRelevanceAI
from .._resource import SyncAPIResource, AsyncAPIResource
from ..types.knowledge import KnowledgeSet, KnowledgeRow

class Knowledge(SyncAPIResource):

    _client: RelevanceAI

    def list_knowledge(
        self, 
    ) -> list[KnowledgeSet]:
        path = "knowledge/sets/list"
        body = {
            "filters": [],
            "sort": [{"update_date":"desc"}]
        }
        response = self._post(path, body=body, cast_to=dict)
        return [KnowledgeSet(**item) for item in response.get("results", [])]
    
    def retrieve_knowledge(
        self, 
        knowledge_set: str, 
        max_results: int = 5
    ) -> list[KnowledgeRow]:
        path = "knowledge/list"
        body = {
            "knowledge_set": knowledge_set,
            "page_size": max_results,
            "sort": [{"insert_date_": "asc"}]
        }
        response = self._post(path, body=body, cast_to=dict)
        return [KnowledgeRow(**item) for item in response.get("results", [])]
        
    def delete_knowledge(
        self,
        knowledge_set: str, 
    ) -> bool:
        path = "knowledge/sets/delete"
        body = {"knowledge_set": knowledge_set}
        response = self._post(path, body=body)
        return response.status_code == 200

class AsyncKnowledge(AsyncAPIResource):

    _client: AsyncRelevanceAI

    async def list_knowledge(
        self, 
    ) -> list[KnowledgeSet]:
        path = "knowledge/sets/list"
        body = {
            "filters": [],
            "sort": [{"update_date":"desc"}]
        }
        response = await self._post(path, body=body, cast_to=dict)
        return [KnowledgeSet(**item) for item in response.get("results", [])]
    
    async def retrieve_knowledge(
        self, 
        knowledge_set: str, 
        max_results: int = 5
    ) -> list[KnowledgeRow]:
        path = "knowledge/list"
        body = {
            "knowledge_set": knowledge_set,
            "page_size": max_results,
            "sort": [{"insert_date_": "asc"}]
        }
        response = await self._post(path, body=body, cast_to=dict)
        return [KnowledgeRow(**item) for item in response.get("results", [])]
        
    async def delete_knowledge(
        self,
        knowledge_set: str, 
    ) -> bool:
        path = "knowledge/sets/delete"
        body = {"knowledge_set": knowledge_set}
        response = await self._post(path, body=body)
        return response.status_code == 200

