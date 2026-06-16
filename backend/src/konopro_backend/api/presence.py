from __future__ import annotations

import time
from threading import Lock

from fastapi import APIRouter, Depends, Request

from konopro_backend.config import BackendSettings
from konopro_backend.dependencies import get_settings
from konopro_backend.schemas import PresenceHeartbeatRequest, PresenceHeartbeatResponse

router = APIRouter(prefix="/v1/presence", tags=["presence"])


@router.post("/heartbeat", response_model=PresenceHeartbeatResponse)
def heartbeat_presence(
    payload: PresenceHeartbeatRequest,
    request: Request,
    settings: BackendSettings = Depends(get_settings),
) -> PresenceHeartbeatResponse:
    now = time.time()
    active_window_s = max(10, int(settings.presence_active_window_s))
    visitors, lock = _presence_registry(request)
    with lock:
        visitors[payload.visitor_id] = now
        cutoff = now - active_window_s
        stale_ids = [visitor_id for visitor_id, seen_at in visitors.items() if seen_at < cutoff]
        for visitor_id in stale_ids:
            visitors.pop(visitor_id, None)
        active_count = len(visitors)

    return PresenceHeartbeatResponse(
        visitor_id=payload.visitor_id,
        active_visitor_count=active_count,
        active_window_s=active_window_s,
    )


def _presence_registry(request: Request) -> tuple[dict[str, float], Lock]:
    state = request.app.state
    if not hasattr(state, "presence_visitors"):
        state.presence_visitors = {}
    if not hasattr(state, "presence_lock"):
        state.presence_lock = Lock()
    return state.presence_visitors, state.presence_lock
