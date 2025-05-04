from __future__ import annotations
import webbrowser

from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.oauth import OAuth

class OAuthManager(SyncAPIResource):

    _client: RelevanceAI

    def add_google_integration(self, auto_open: bool = False) -> str:
        path = "auth/oauth/get_url"
        body = {"provider": "google", "types": ["email-read-write"], "redirect_url": ""}
        response = self._post(path, body=body, cast_to=dict)
        response = OAuth(**response)
        if auto_open:
            webbrowser.open(response.auth_url)
        return response.auth_url

    def list_active_integrations(self) -> dict:
        path = "auth/oauth/accounts/list"
        response = self._post(path, cast_to=dict)
        return response
