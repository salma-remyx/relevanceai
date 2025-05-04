from __future__ import annotations
import time
from typing import TYPE_CHECKING, Type, TypeVar, Any, Optional, Union
import httpx
import asyncio

if TYPE_CHECKING:
    from ._client import RelevanceAI, AsyncRelevanceAI

ResponseT = TypeVar('ResponseT')

class SyncAPIResource:
    _client: RelevanceAI
    
    def __init__(self, client: RelevanceAI) -> None:
        self._client = client

    def _get(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[dict[str, Any]] = None, 
        params: Optional[dict[str, Any]] = None
    ) -> ResponseT:
        response = self._client.get(path, json=body, params=params)
        return self._cast_response(response, cast_to)

    def _post(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[dict[str, Any]] = None, 
        params: Optional[dict[str, Any]] = None, 
        options: Optional[dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = self._client.post(path, json=body, params=params, **options)
        return self._cast_response(response, cast_to)

    def _patch(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[dict[str, Any]] = None, 
        params: Optional[dict[str, Any]] = None, 
        options: Optional[dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = self._client.patch(path, json=body, params=params, **options)
        return self._cast_response(response, cast_to)

    def _put(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[dict[str, Any]] = None, 
        params: Optional[dict[str, Any]] = None, 
        options: Optional[dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = self._client.put(path, json=body, params=params, **options)
        return self._cast_response(response, cast_to)

    def _delete(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[dict[str, Any]] = None, 
        params: Optional[dict[str, Any]] = None, 
        options: Optional[dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = self._client.delete(path, json=body, params=params, **options)
        return self._cast_response(response, cast_to)

    def _cast_response(
        self, 
        response: httpx.Response, 
        cast_to: Optional[Type[ResponseT]] = None
    ) -> Union[ResponseT, dict, httpx.Response]:
        if not cast_to:
            return response
        if isinstance(response, httpx.Response):
            response = response.json()
        if cast_to == dict:
            return response
        return cast_to(response)

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

class AsyncAPIResource:
    _client: AsyncRelevanceAI
    
    def __init__(self, client: AsyncRelevanceAI) -> None:
        self._client = client

    async def _get(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[dict[str, Any]] = None, 
        params: Optional[dict[str, Any]] = None
    ) -> ResponseT:
        response = await self._client.get(path, json=body, params=params)
        return await self._cast_response(response, cast_to)

    async def _post(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[dict[str, Any]] = None, 
        params: Optional[dict[str, Any]] = None, 
        options: Optional[dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = await self._client.post(path, json=body, params=params, **options)
        return await self._cast_response(response, cast_to)

    async def _patch(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[dict[str, Any]] = None, 
        params: Optional[dict[str, Any]] = None, 
        options: Optional[dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = await self._client.patch(path, json=body, params=params, **options)
        return await self._cast_response(response, cast_to)

    async def _put(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[dict[str, Any]] = None, 
        params: Optional[dict[str, Any]] = None, 
        options: Optional[dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = await self._client.put(path, json=body, params=params, **options)
        return await self._cast_response(response, cast_to)

    async def _delete(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[dict[str, Any]] = None, 
        params: Optional[dict[str, Any]] = None, 
        options: Optional[dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = await self._client.delete(path, json=body, params=params, **options)
        return await self._cast_response(response, cast_to)

    async def _cast_response(
        self, 
        response: httpx.Response, 
        cast_to: Optional[Type[ResponseT]] = None
    ) -> Union[ResponseT, dict, httpx.Response]:
        if not cast_to:
            return response
        if isinstance(response, httpx.Response):
            response = response.json()
        if cast_to == dict:
            return response
        return cast_to(response)

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)