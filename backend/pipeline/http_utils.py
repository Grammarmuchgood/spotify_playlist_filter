from __future__ import annotations

import time
from typing import Callable

import requests


def make_throttle(min_interval_seconds: float) -> Callable[[], None]:
    """Returns a rate limiter that enforces a minimum spacing between
    calls, remembering "when did I last run" in its own closure - each
    API gets an independent limiter this way, without a separate
    module-level variable and near-duplicate function per API."""
    last_call = 0.0

    def throttle() -> None:
        nonlocal last_call
        # time.monotonic() only ever moves forward (unaffected by system
        # clock changes), which matters for measuring elapsed durations.
        elapsed = time.monotonic() - last_call
        if elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)
        last_call = time.monotonic()

    return throttle


def get_with_retry(throttle_fn, url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 15) -> requests.Response:
    """Shared by every external API this pipeline calls (iTunes, MusicBrainz,
    lyrics.ovh): a transient failure (5xx, connection reset, timeout, DNS
    blip - all confirmed to happen in practice on this network, not
    hypothetical) shouldn't kill a 30-60 minute rate-limited batch run.
    Retries with a growing delay (1s, 2s, 4s, 8s) before finally giving up
    and raising.

    Deliberately does NOT call raise_for_status() itself - a "normal" error
    status (like lyrics.ovh's 404 for "no lyrics found") is a legitimate,
    meaningful response for some APIs, not a failure to retry. Only 5xx
    (server-side, likely transient) is retried here; the caller decides
    what any other status code (2xx, 404, etc.) means for their API.
    """
    last_exc = None
    resp = None
    for attempt in range(4):
        throttle_fn()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            # Raised before any response comes back at all (timeout,
            # connection reset, DNS/routing blip) - there's no status_code
            # to check here, so this is caught separately from a 5xx.
            last_exc = exc
            time.sleep(2**attempt)
            continue
        if resp.status_code < 500:
            return resp
        time.sleep(2**attempt)
    if last_exc is not None:
        raise last_exc
    return resp
