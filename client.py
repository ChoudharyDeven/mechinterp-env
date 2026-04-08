"""
MechInterpEnv — Python client for connecting to the mechinterp-env server.

Usage (async):
    async with MechInterpEnv(base_url="https://your-space.hf.space") as env:
        obs = await env.reset(task_id="head-identification")
        obs = await env.step(MechInterpAction(action_type="ablate_head", layer=0, head=2))

Usage (sync):
    with MechInterpEnv(base_url="https://your-space.hf.space").sync() as env:
        obs = env.reset(task_id="head-identification")
        obs = env.step(MechInterpAction(action_type="ablate_head", layer=0, head=2))
"""

from __future__ import annotations
import asyncio
import json
import httpx
from typing import Optional
from models import MechInterpAction, MechInterpObservation


class MechInterpEnv:
    """
    Async HTTP client for mechinterp-env.
    Compatible with the OpenEnv client interface.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def reset(self, task_id: str = "head-identification") -> MechInterpObservation:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        resp = await self._client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return MechInterpObservation(**resp.json())

    async def step(self, action: MechInterpAction) -> MechInterpObservation:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        resp = await self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return MechInterpObservation(**resp.json())

    async def state(self) -> dict:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def sync(self) -> "SyncMechInterpEnv":
        """Return a synchronous wrapper around this client."""
        return SyncMechInterpEnv(self.base_url)


class SyncMechInterpEnv:
    """Synchronous wrapper around MechInterpEnv for non-async code."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client  = httpx.Client(base_url=self.base_url, timeout=30.0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    def reset(self, task_id: str = "head-identification") -> MechInterpObservation:
        resp = self._client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return MechInterpObservation(**resp.json())

    def step(self, action: MechInterpAction) -> MechInterpObservation:
        resp = self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return MechInterpObservation(**resp.json())

    def state(self) -> dict:
        resp = self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()
