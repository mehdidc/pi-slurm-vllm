import contextlib
import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

import deploy_opentela as deploy


PEER = "QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7"


class ArgumentTests(unittest.TestCase):
    def parse(self, *args):
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()):
            return deploy.parse_args(["--model", "Test-Model", *args])

    def test_gateway_urls_accept_origins_and_service_urls(self):
        for origin in ("https://opentela-laion-vm.taild3d308.ts.net",
                       "https://api.opentela.ai", "http://localhost:8092",
                       "http://127.0.0.1:8092", "http://[::1]:8092"):
            for suffix in ("", "/", "/v1/service/llm", "/v1/service/llm/",
                           "/v1/service/llm/v1", "/v1/service/llm/v1/"):
                with self.subTest(url=origin + suffix):
                    args = self.parse("--api-url", origin + suffix, "--api-key", "key")
                    expected = origin + "/v1/service/llm/v1" if suffix.strip("/") else origin
                    self.assertEqual(args.api_url, expected)

    def test_all_gateways_require_key(self):
        for url in ("https://api.opentela.ai", "https://opentela-laion-vm.taild3d308.ts.net",
                    "http://localhost:8092"):
            with self.subTest(url=url), self.assertRaises(SystemExit):
                self.parse("--api-url", url)

    def test_local_gateway_accepts_key_and_rejects_invalid_key(self):
        args = self.parse("--api-url", "http://localhost:8092", "--api-key", "local-key")
        self.assertEqual(args.api_key, "local-key")
        with self.assertRaises(SystemExit):
            self.parse("--api-url", "http://localhost:8092", "--api-key", "bad\nkey")

    def test_environment_key_is_preserved(self):
        with patch.dict(os.environ, {"OPENTELA_API_KEY": "env-key"}, clear=True):
            args = deploy.parse_args(["--model", "Test-Model", "--api-url", "http://localhost:8092"])
        self.assertEqual(args.api_key, "env-key")

    def test_remote_http_and_invalid_origins_still_rejected(self):
        for url in ("http://gateway.example", "http://localhost:8092/v1",
                    "http://user@localhost:8092", "http://localhost:8092?x=1",
                    "https://gateway.example/v1/service/other",
                    "https://gateway.example/v1/service/llm#fragment"):
            with self.subTest(url=url), self.assertRaises(SystemExit):
                self.parse("--api-url", url, "--api-key", "key")

    def test_error_redacts_api_key(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(deploy.signal, "signal"), \
                patch.object(deploy.slurm, "supported_runner_path", side_effect=ValueError("failure: secret-key")), \
                contextlib.redirect_stderr(output):
            status = deploy.main(["--model", "Test-Model", "--cluster", "test",
                                  "--api-url", "http://localhost:8092", "--api-key", "secret-key"])
        self.assertEqual(status, 1)
        self.assertEqual(output.getvalue(), "Error: failure: [redacted]\n")


class LocalRouteTests(unittest.TestCase):
    def test_gateway_exposing_only_service_route(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                if self.path != "/v1/service/llm/v1/models":
                    self.send_error(404)
                elif self.headers.get("Authorization") != "Bearer gateway-key":
                    self.send_error(401)
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"data":[{"id":"Test-Model"}]}')

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                origin = f"http://127.0.0.1:{server.server_port}"
                args = deploy.parse_args(["--model", "Test-Model", "--api-key", "gateway-key",
                                          "--api-url", origin + "/v1/service/llm",
                                          "--opentela-timeout", "1"])
                result = deploy.wait_for_route(args, Mock(poll=lambda: None), origin, "Test-Model")
                self.assertEqual(result, origin + "/v1/service/llm/v1")
                self.assertEqual(requests, ["/v1/service/llm/v1/models"])
            finally:
                server.shutdown()
                thread.join()

    def test_self_hosted_service_url_preserves_route_and_key(self):
        origin = "https://opentela-laion-vm.taild3d308.ts.net"
        with patch.dict(os.environ, {}, clear=True):
            args = deploy.parse_args(["--model", "Test-Model", "--api-key", "self-hosted-key",
                                      "--api-url", origin + "/v1/service/llm"])
        local_url = "http://127.0.0.1:12345"
        with patch.object(deploy, "get_json", return_value={"data": [{"id": "Test-Model"}]}) as get_json:
            result = deploy.wait_for_route(args, Mock(poll=lambda: None), local_url, "Test-Model")
        self.assertEqual(result, f"{origin}/v1/service/llm/v1")
        get_json.assert_called_once_with(result + "/models", "self-hosted-key", use_proxy=True)

    def test_route_failure_reports_url_and_http_status_before_timeout(self):
        url = "https://gateway.example/v1/service/llm/v1"
        args = SimpleNamespace(api_url=url, api_key="secret-key", opentela_timeout=2, poll_interval=1)
        failure = deploy.HTTPError(url, 404, "secret-key in remote error", {}, None)
        output = io.StringIO()
        with patch.object(deploy, "get_json", side_effect=failure), \
                patch.object(deploy.time, "monotonic", side_effect=[0, 0, 0, 0, 3]), \
                patch.object(deploy.time, "sleep"), contextlib.redirect_stderr(output):
            with self.assertRaisesRegex(TimeoutError, "HTTP 404"):
                deploy.wait_for_route(args, Mock(poll=lambda: None), "http://localhost:1234", "Test-Model")
        self.assertIn(url + "/models: HTTP 404", output.getvalue())
        self.assertNotIn(args.api_key, output.getvalue())

    def test_local_gateway_bypasses_proxies_and_receives_key(self):
        api_key = "local-key"
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append((self.path, self.headers.get("Authorization")))
                body = ({"id": PEER} if self.path == "/v1/self"
                        else {"data": [{"id": "Test-Model"}]})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                args = SimpleNamespace(api_url=url, api_key=api_key,
                                       opentela_timeout=5, poll_interval=1)
                with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1",
                                             "no_proxy": ""}, clear=True):
                    result = deploy.wait_for_route(args, Mock(poll=lambda: None),
                                                   url, "Test-Model")
                self.assertEqual(result, f"{url}/v1/p2p-service/{PEER}/llm/v1")
                self.assertEqual(requests, [
                    ("/v1/self", None),
                    (f"/v1/p2p-service/{PEER}/llm/v1/models",
                     f"Bearer {api_key}" if api_key else None),
                ])
            finally:
                server.shutdown()
                thread.join()

    def test_authentication_failure_does_not_retry_or_expose_key(self):
        args = SimpleNamespace(api_url="http://localhost:8092", api_key="local-key",
                               opentela_timeout=5, poll_interval=1)
        failure = deploy.HTTPError(args.api_url, 401, "Unauthorized", {}, None)
        with patch.object(deploy, "get_json", side_effect=[{"id": PEER}, failure]):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
                deploy.wait_for_route(args, Mock(poll=lambda: None), args.api_url, "Test-Model")
        self.assertNotIn(args.api_key, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
