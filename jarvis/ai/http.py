"""Tiny stdlib HTTP transport used by AI providers.

Why this exists: providers must not leak `urllib` details into the rest of
JARVIS, and tests need a seam that never touches the network. The transport
is injected, so every provider can be exercised with a fake.

Errors are reported as `TransportError` with a coarse `kind`; providers map
those kinds to their own typed errors.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

KIND_TIMEOUT = "timeout"
KIND_CONNECTION = "connection"
KIND_AUTH = "auth"
KIND_RATE_LIMIT = "rate_limit"
KIND_HTTP = "http"
KIND_PARSE = "parse"


class TransportError(Exception):
    """Transport-level failure, before any provider interpretation."""

    def __init__(self, kind: str, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.raw_message = message


class HttpTransport(Protocol):
    """Minimal JSON-over-HTTP surface needed by the providers."""

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Any: ...

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Any: ...

    def post_stream_lines(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Iterator[str]: ...


class UrllibTransport:
    """Default transport built on `urllib.request` (standard library)."""

    def __init__(self, opener: Any | None = None) -> None:
        self._opener = opener or urllib.request.urlopen

    def get_json(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        body = self._request(url, data=None, headers=headers, timeout=timeout)
        return _parse_json(body)

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        body = self._request(url, data=payload, headers=headers, timeout=timeout)
        return _parse_json(body)

    def post_stream_lines(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Iterator[str]:
        response = self._open(url, data=payload, headers=headers, timeout=timeout)
        try:
            with response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line:
                        yield line
        except TransportError:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise TransportError(KIND_TIMEOUT, f"stream timed out: {exc}") from exc
        except OSError as exc:
            raise TransportError(KIND_CONNECTION, f"stream failed: {exc}") from exc

    def _request(
        self,
        url: str,
        data: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> bytes:
        response = self._open(url, data=data, headers=headers, timeout=timeout)
        try:
            with response:
                return response.read()
        except (TimeoutError, socket.timeout) as exc:
            raise TransportError(KIND_TIMEOUT, f"read timed out after {timeout}s") from exc
        except OSError as exc:
            raise TransportError(KIND_CONNECTION, f"read failed: {exc}") from exc

    def _open(
        self,
        url: str,
        data: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> Any:
        body = None
        method = "GET"
        if data is not None:
            method = "POST"
            body = json.dumps(data).encode("utf-8")
        request_headers = {"Accept": "application/json"}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            url, data=body, headers=request_headers, method=method
        )
        try:
            return self._opener(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = _read_error_body(exc)
            raise TransportError(
                _kind_for_status(exc.code),
                f"HTTP {exc.code}: {detail}",
                status=exc.code,
            ) from None
        except (TimeoutError, socket.timeout) as exc:
            raise TransportError(
                KIND_TIMEOUT, f"request timed out after {timeout}s"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise TransportError(
                KIND_CONNECTION, f"cannot reach {url}: {reason}"
            ) from exc
        except OSError as exc:
            raise TransportError(KIND_CONNECTION, f"cannot reach {url}: {exc}") from exc


def _kind_for_status(status: int) -> str:
    if status in (401, 403):
        return KIND_AUTH
    if status == 429:
        return KIND_RATE_LIMIT
    return KIND_HTTP


def _read_error_body(error: urllib.error.HTTPError) -> str:
    try:
        raw = error.read()
    except Exception:  # pragma: no cover - defensive
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return raw.strip()[:300]


def _parse_json(body: bytes | str) -> Any:
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TransportError(KIND_PARSE, f"malformed JSON response: {exc}") from exc
