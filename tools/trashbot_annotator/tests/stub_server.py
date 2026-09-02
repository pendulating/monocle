"""Stub vLLM OpenAI-compatible server for tests.

The server answers POST /v1/chat/completions with a canned chat
completion. The message content holds one valid annotation JSON
object. The handler records each request body and counts the image_url
content parts it saw.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from trashbot_annotator.taxonomy import NORM_DIMENSIONS

# One valid annotation payload. Tests parse this from the response.
CANNED_ANNOTATION: dict[str, object] = {
    "interactions": [
        {
            "category": "instruction",
            "address": "to_robot",
            "who": "caregiver",
            "evidence": "Parent says: push it.",
        },
        {
            "category": "gossip",
            "address": "about_robot",
            "who": "bystander",
            "evidence": "They discuss the robot.",
        },
    ],
    "norm_dimensions": [
        {"dimension": NORM_DIMENSIONS[0], "relevance": 2, "evidence": "Child smiles at it."},
        {"dimension": NORM_DIMENSIONS[8], "relevance": 3, "evidence": "People yield the path."},
    ],
    "proxemic_events": [
        {
            "event": "trash_deposit",
            "who": "child",
            "distance_m": 0.3,
            "zone": "intimate",
            "evidence": "The child drops a cup into the robot.",
        },
        {
            "event": "gaze_at_robot",
            "who": "runner",
            "distance_m": 5.0,
            "zone": "public",
            "evidence": "A runner turns the head toward the robot.",
        },
    ],
    "summary": "A family guides the robot and talks about it.",
}


class _StubHandler(BaseHTTPRequestHandler):
    """Handle one chat completion request."""

    server_version = "StubVLLM/1.0"

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """Serve the chat completion endpoint."""
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": f"unknown path {self.path}"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "bad json body"})
            return

        server: StubVLLMServer = self.server  # type: ignore[assignment]
        server.requests.append(body)
        server.image_part_counts.append(count_image_url_parts(body))

        payload = server.response_payload

        completion = {
            "id": "chatcmpl-stub-0001",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "stub-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(payload),
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        self._send_json(200, completion)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence per-request log lines."""


def count_image_url_parts(body: dict[str, object]) -> int:
    """Count image_url content parts across all messages."""
    total = 0
    for message in body.get("messages", []):  # type: ignore[union-attr]
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                total += 1
    return total


class StubVLLMServer(ThreadingHTTPServer):
    """Threaded HTTP server that records chat completion requests."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, response_payload: dict[str, object] | None = None) -> None:
        super().__init__(("127.0.0.1", 0), _StubHandler)
        self.requests: list[dict[str, object]] = []
        self.image_part_counts: list[int] = []
        self.response_payload: dict[str, object] = (
            response_payload if response_payload is not None else CANNED_ANNOTATION
        )

    @property
    def base_url(self) -> str:
        """Return the base URL of the server."""
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"


def start_stub_server(
    response_payload: dict[str, object] | None = None,
) -> tuple[StubVLLMServer, threading.Thread]:
    """Start the stub server in a background thread."""
    server = StubVLLMServer(response_payload=response_payload)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_stub_server(server: StubVLLMServer, thread: threading.Thread) -> None:
    """Stop the stub server and join its thread."""
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
