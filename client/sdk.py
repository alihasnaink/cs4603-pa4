"""Python client SDK for the deployed Document Analyst (Part 3).

TODO: Implement `DocumentAnalystClient` and `AnalystClientError` per Task 3.1:
  - __init__(endpoint_name, host=None, token=None, timeout=120.0, max_retries=3):
    read DATABRICKS_HOST/DATABRICKS_TOKEN from env when not provided.
  - ask(question) -> str
  - ask_streaming(question) -> Iterator[str]   (yield chunks as they arrive)
  - health_check() -> bool                      (True only when endpoint READY)
  - exponential backoff on 429/503, TimeoutError with elapsed time, and wrap HTTP
    errors in AnalystClientError(status_code, message, request_id).
"""

from __future__ import annotations

from collections.abc import Iterator


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        import os
        self.endpoint_name = endpoint_name
        self.host = host or os.environ.get("DATABRICKS_HOST")
        self.token = token or os.environ.get("DATABRICKS_TOKEN")
        
        if not self.host or not self.token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be provided or set in environment variables.")
        
        self.host = self.host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _request_with_retry(self, method: str, url: str, stream: bool = False, **kwargs) -> __import__("requests").Response:
        import time

        import requests
        retries = 0
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= self.timeout:
                raise TimeoutError(f"Request timed out after {elapsed:.2f} seconds.")
            
            remaining_timeout = max(0.1, self.timeout - elapsed)
            try:
                response = requests.request(
                    method, url, stream=stream, timeout=remaining_timeout, **kwargs
                )
            except requests.exceptions.Timeout:
                elapsed = time.time() - start_time
                raise TimeoutError(f"Request timed out after {elapsed:.2f} seconds.") from None
            
            if response.status_code in (429, 503):
                if retries >= self.max_retries:
                    raise AnalystClientError(
                        f"Max retries reached. Last error: {response.text}",
                        status_code=response.status_code,
                        request_id=response.headers.get("x-request-id")
                    )
                
                sleep_time = 2 ** retries
                if time.time() - start_time + sleep_time >= self.timeout:
                    raise TimeoutError(f"Request timed out after {time.time() - start_time:.2f} seconds.")
                    
                time.sleep(sleep_time)
                retries += 1
                continue
                
            if not response.ok:
                raise AnalystClientError(
                    f"Request failed: {response.text}",
                    status_code=response.status_code,
                    request_id=response.headers.get("x-request-id")
                )
            
            return response

    def health_check(self) -> bool:
        url = f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = self._request_with_retry("GET", url, headers=headers)
            data = response.json()
            return data.get("state", {}).get("ready") == "READY"
        except TimeoutError:
            raise
        except Exception:
            return False

    def ask(self, question: str) -> str:
        if not self.health_check():
            raise AnalystClientError("Endpoint is not READY")
            
        url = f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        payload = {"messages": [{"role": "user", "content": question}]}
        
        response = self._request_with_retry("POST", url, headers=headers, json=payload)
        data = response.json()
        
        try:
            return data[0]["messages"][-1]["content"]
        except (KeyError, IndexError, TypeError):
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return str(data)

    def ask_streaming(self, question: str) -> Iterator[str]:
        import json
        if not self.health_check():
            raise AnalystClientError("Endpoint is not READY")
            
        url = f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        payload = {"messages": [{"role": "user", "content": question}]}
        response = self._request_with_retry("POST", url, headers=headers, json=payload, stream=True)
        
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                            delta_content = chunk["choices"][0]["delta"].get("content", "")
                            if delta_content:
                                yield delta_content
                        elif isinstance(chunk, list) and len(chunk) > 0 and "messages" in chunk[0]:
                            yield chunk[0]["messages"][-1]["content"]
                    except json.JSONDecodeError:
                        pass
        else:
            data = response.json()
            try:
                yield data[0]["messages"][-1]["content"]
            except (KeyError, IndexError, TypeError):
                try:
                    yield data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    yield str(data)
