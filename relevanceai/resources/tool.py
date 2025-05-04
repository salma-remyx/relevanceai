from __future__ import annotations
from typing import List, Optional, Dict

from .._client import RelevanceAI, AsyncRelevanceAI
from .._resource import SyncAPIResource, AsyncAPIResource
from ..types.params import ParamsBase
from ..types.transformations import TransformationBase


class Tool(SyncAPIResource):
    _client: RelevanceAI

    def __init__(self, client: RelevanceAI, tool_id: str, **kwargs):
        super().__init__(client=client)
        self.tool_id = tool_id

    def update(self, updates: dict, partial_update: bool = True) -> dict:
        path = "studios/bulk_update"
        body = {
            "partial_update": partial_update,
            "updates": [updates | {"studio_id": self.tool_id}],
        }
        return self._post(path, body=body, cast_to=dict)

    def trigger(self, params: dict | None = None):
        path = f"studios/{self.tool_id}/trigger_limited"
        body = {"params": params, "project": self._client.project}
        return self._post(path=path, body=body, cast_to=dict)

    def get_params_schema(self) -> str:
        response = self._get(f"studios/{self.tool_id}/get", cast_to=dict)
        params_schema = response["studio"]["params_schema"]["properties"]
        return params_schema

    def get_transformations_schema(self) -> str:
        response = self._get(f"studios/{self.tool_id}/get", cast_to=dict)
        steps_schema = response["studio"]["transformations"]["steps"]
        return steps_schema

    def update_metadata(
        self,
        title: str | None = None,
        description: str | None = None,
        public: bool | None = None,
    ):
        response = self._get(f"studios/{self.tool_id}/get", cast_to=dict)

        updates = {
            "studio_id": self.tool_id,
            "title": title or response["studio"].get("title"),
            "description": description or response["studio"].get("description"),
            "public": public or response["studio"].get("public"),
        }

        path = "studios/bulk_update"
        body = {"updates": [updates], "partial_update": True}
        return self._post(path, body=body, cast_to=dict)

    def update_params(self, params: Dict[str, ParamsBase]) -> Tool:
        params_schema = {"properties": {}, "required": [], "type": "object"}

        param_values = {field_name: param.value for field_name, param in params.items() if param.value is not None}

        for field_name, param in params.items():
            param_dict = param.model_dump(exclude_none=True)
            params_schema["properties"][field_name] = param_dict
            if param.required:
                params_schema["required"].append(field_name)

        state_mapping = {field_name: f"params.{field_name}" for field_name in params.keys()}

        path = "studios/bulk_update"
        body = {
            "updates": [
                {
                    "studio_id": self.tool_id,
                    "params": param_values,
                    "params_schema": params_schema,
                    "state_mapping": state_mapping,
                }
            ],
            "partial_update": True,
        }

        return self._post(path, body=body, cast_to=dict)

    def update_transformations(self, transformations: List[TransformationBase]) -> dict:
        response = self._get(f"studios/{self.tool_id}/get", cast_to=dict)
        current_state = response["studio"].get("state_mapping", {})

        state_mapping = {**current_state, **{step.name: f"steps.{step.name}.output" for step in transformations}}

        transformation_config = {"steps": [transform.model_dump(exclude_none=True) for transform in transformations]}

        path = "studios/bulk_update"
        body = {
            "updates": [
                {"studio_id": self.tool_id, "transformations": transformation_config, "state_mapping": state_mapping}
            ],
            "partial_update": True,
        }

        return self._post(path, body=body, cast_to=dict)

    def update_outputs(
        self,
        last_step: bool = True,
        output_mapping: Optional[dict] = None,
        output_schema_properties: Optional[dict] = None,
    ) -> dict:
        response = self._get(f"studios/{self.tool_id}/get", cast_to=dict)
        current_steps = response["studio"].get("transformations", {}).get("steps", [])

        transformations = {"steps": current_steps}
        transformations["output"] = output_mapping if not last_step else None

        updates = {
            "studio_id": self.tool_id,
            "transformations": transformations,
        }

        if output_schema_properties:
            updates["output_schema"] = {"properties": output_schema_properties}

        path = "studios/bulk_update"
        body = {"updates": [updates], "partial_update": True}

        return self._post(path, body=body, cast_to=dict)

    def get_link(self):
        return f"https://app.relevanceai.com/agents/{self._client.region}/{self._client.project}/{self.tool_id}"

    def __repr__(self):
        return f'Tool(tool_id="{self.tool_id}")'


class AsyncTool(AsyncAPIResource):
    _client: AsyncRelevanceAI

    def __init__(self, client: AsyncRelevanceAI, tool_id: str, **kwargs):
        super().__init__(client=client)
        self.tool_id = tool_id

    async def update(self, updates: dict, partial_update: Optional[bool] = True) -> dict:
        path = "studios/bulk_update"
        body = {
            "partial_update": partial_update,
            "updates": [updates | {"studio_id": self.tool_id}],
        }
        response = await self._post(path, body=body, cast_to=dict)
        return response

    async def trigger(self, params: dict | None = None):
        path = f"studios/{self.tool_id}/trigger_limited"
        body = {"params": params, "project": self._client.project}
        response = await self._post(path=path, body=body, cast_to=dict)
        return response

    async def get_params_schema(self) -> dict:
        response = await self._get(f"studios/{self.tool_id}/get", cast_to=dict)
        params_schema = response["studio"]["params_schema"]["properties"]
        return params_schema

    async def get_transformations_schema(self) -> dict:
        response = await self._get(f"studios/{self.tool_id}/get", cast_to=dict)
        steps_schema = response["studio"]["transformations"]["steps"]
        return steps_schema

    async def update_metadata(
        self,
        title: str | None = None,
        description: str | None = None,
        public: bool | None = None,
    ):
        response = await self._get(f"studios/{self.tool_id}/get", cast_to=dict)

        updates = {
            "studio_id": self.tool_id,
            "title": title | response["studio"].get("title"),
            "description": description | response["studio"].get("description"),
            "public": public | response["studio"].get("public"),
        }

        path = "studios/bulk_update"
        body = {"updates": [updates], "partial_update": True}
        response = await self._post(path, body=body, cast_to=dict)
        return response

    async def update_params(self, params: Dict[str, ParamsBase]) -> dict:
        params_schema = {"properties": {}, "required": [], "type": "object"}

        param_values = {field_name: param.value for field_name, param in params.items() if param.value is not None}

        for field_name, param in params.items():
            param_dict = param.model_dump(exclude_none=True)
            params_schema["properties"][field_name] = param_dict
            if param.required:
                params_schema["required"].append(field_name)

        state_mapping = {field_name: f"params.{field_name}" for field_name in params.keys()}

        path = "studios/bulk_update"
        body = {
            "updates": [
                {
                    "studio_id": self.tool_id,
                    "params": param_values,
                    "params_schema": params_schema,
                    "state_mapping": state_mapping,
                }
            ],
            "partial_update": True,
        }

        response = await self._post(path, body=body, cast_to=dict)
        return response

    async def update_transformations(self, transformations: List[TransformationBase]) -> dict:
        response = await self._get(f"studios/{self.tool_id}/get", cast_to=dict)
        current_state = response["studio"].get("state_mapping", {})

        state_mapping = {**current_state, **{step.name: f"steps.{step.name}.output" for step in transformations}}

        transformation_config = {"steps": [transform.model_dump(exclude_none=True) for transform in transformations]}

        path = "studios/bulk_update"
        body = {
            "updates": [
                {"studio_id": self.tool_id, "transformations": transformation_config, "state_mapping": state_mapping}
            ],
            "partial_update": True,
        }

        response = await self._post(path, body=body, cast_to=dict)
        return response

    async def update_outputs(
        self,
        last_step: bool = True,
        output_mapping: Optional[dict] = None,
        output_schema_properties: Optional[dict] = None,
    ) -> dict:
        response = await self._get(f"studios/{self.tool_id}/get", cast_to=dict)
        current_steps = response["studio"].get("transformations", {}).get("steps", [])

        transformations = {"steps": current_steps}
        transformations["output"] = output_mapping if not last_step else None

        updates = {
            "studio_id": self.tool_id,
            "transformations": transformations,
        }

        if output_schema_properties:
            updates["output_schema"] = {"properties": output_schema_properties}

        path = "studios/bulk_update"
        body = {"updates": [updates], "partial_update": True}

        response = await self._post(path, body=body, cast_to=dict)
        return response

    def get_link(self):
        return f"https://app.relevanceai.com/agents/{self._client.region}/{self._client.project}/{self.tool_id}"

    def __repr__(self):
        return f'AsyncTool(tool_id="{self.tool_id}")'
