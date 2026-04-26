"""调用示例（开两个终端，一个跑 main.py，另一个跑这个）。"""
import sys

import httpx


def main():
    base = "http://127.0.0.1:8000"

    print("--- /health ---")
    print(httpx.get(f"{base}/health").json())

    print("\n--- /chat ---")
    r = httpx.post(f"{base}/chat", json={"message": "查 NVDA 现价"}, timeout=60)
    print(r.json())

    print("\n--- /chat/stream ---")
    with httpx.stream("POST", f"{base}/chat/stream", json={"message": "查 AAPL"}, timeout=60) as resp:
        for line in resp.iter_lines():
            if line:
                print(line)


if __name__ == "__main__":
    sys.exit(main())
