from __future__ import annotations
import json
import uuid
from typing import List, Optional

from .._client import RelevanceAI, AsyncRelevanceAI
from .._resource import SyncAPIResource, AsyncAPIResource
from ..resources.tool import Tool, AsyncTool


class ToolsManager(SyncAPIResource):

    _client: RelevanceAI

    def list_tools(
        self,
        max_results: Optional[int] = 100,
    ) -> List[Tool]:
        path = "studios/list"
        params = {
            "filters": json.dumps(
                [
                    {
                        "filter_type": "exact_match",
                        "field": "project",
                        "condition_value": self._client.project,
                        "condition": "==",
                    }
                ]
            ),
            "page_size": max_results,
        }
        response = self._client.get(path, params=params, cast_to=dict)
        tools = [Tool(client=self._client, tool_id=item.get('studio_id')) for item in response.get("results", [])]
        return tools
    
    def retrieve_tool(self, tool_id: str) -> Tool:
        path = f"studios/{tool_id}/get"
        response = self._get(path, cast_to=dict)
        return Tool(client=self._client, tool_id=response["studio"].get("studio_id"))
    
    def create_tool(
        self, 
        title: str, 
        description: str, 
        public: bool = False,
        params_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
        transformations: Optional[dict] = None
    ) -> Tool:
        tool_id = str(uuid.uuid4())
        path = "studios/bulk_update"
        body = {
            "updates": [
                {
                    "title": title,
                    "public": public,
                    "project": self._client.project,
                    "description": description,
                    "version": "latest",
                    "params_schema": params_schema or {"properties": {}, "required": [], "type": "object"},
                    "output_schema": output_schema or {},
                    "transformations": transformations or {"steps": []},
                    "studio_id": tool_id
                }
            ],
            "partial_update": True
        }
        _ = self._post(path, body=body)
        tool = self.retrieve_tool(tool_id)
        return tool

    def clone_tool(
        self,
        tool_id,
    ) -> Optional[Tool]: 
        path = "/studios/clone"
        body = {
            "studio_id": tool_id,
            "project": self._client.project,
            "region": self._client.region
        }
        response = self._post(path, body=body, cast_to=dict)
        cloned_tool_id = response.get("studio_id", None)
        if cloned_tool_id: 
            tool = self.retrieve_tool(cloned_tool_id)
            return tool
        else:
            return False

    def delete_tool(self, tool_id: str) -> bool:
        path = "studios/bulk_delete"
        body = {"ids": [tool_id]}
        response = self._post(path, body=body)
        return response.status_code == 200


class AsyncToolsManager(AsyncAPIResource):

    _client: AsyncRelevanceAI

    async def list_tools(self, max_results: Optional[int] = 100) -> List[AsyncTool]:
        path = "studios/list"
        params = {
            "filters": json.dumps(
                [
                    {
                        "filter_type": "exact_match",
                        "field": "project",
                        "condition_value": self._client.project,
                        "condition": "==",
                    }
                ]
            ),
            "page_size": max_results,
        }
        response = await self._client.get(path, params=params, cast_to=dict)
        tools = [AsyncTool(client=self._client, tool_id=item.get('studio_id')) for item in response.get("results", [])]
        return tools
    
    async def retrieve_tool(self, tool_id: str) -> AsyncTool:
        path = f"studios/{tool_id}/get"
        response = await self._get(path, cast_to=dict)
        return AsyncTool(client=self._client, tool_id=response["studio"].get("studio_id"))
    
    async def create_tool(
        self, 
        title: str, 
        description: str, 
        public: bool = False,
        params_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
        transformations: Optional[dict] = None
    ) -> AsyncTool:
        tool_id = str(uuid.uuid4())
        path = "studios/bulk_update"
        body = {
            "updates": [
                {
                    "title": title,
                    "public": public,
                    "project": self._client.project,
                    "description": description,
                    "version": "latest",
                    "params_schema": params_schema or {"properties": {}, "required": [], "type": "object"},
                    "output_schema": output_schema or {},
                    "transformations": transformations or {"steps": []},
                    "studio_id": tool_id
                }
            ],
            "partial_update": True
        }
        _ = await self._post(path, body=body)
        tool = await self.retrieve_tool(tool_id)
        return tool

    async def clone_tool(self, tool_id: str) -> Optional[AsyncTool]:
        path = "/studios/clone"
        body = {
            "studio_id": tool_id,
            "project": self._client.project,
            "region": self._client.region
        }
        response = await self._post(path, body=body, cast_to=dict)
        cloned_tool_id = response.get("studio_id", None)
        if cloned_tool_id:
            tool = await self.retrieve_tool(cloned_tool_id)
            return tool
        return False

    async def delete_tool(self, tool_id: str) -> bool:
        path = "studios/bulk_delete"
        body = {"ids": [tool_id]}
        response = await self._post(path, body=body)
        return response.status_code == 200
