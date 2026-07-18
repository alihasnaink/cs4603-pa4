import json

with open("pa4.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        
        if "# from client.sdk import DocumentAnalystClient" in src:
            cell["source"] = [
                "import os\n",
                "from client.sdk import DocumentAnalystClient\n",
                "\n",
                "user = os.environ.get('USER', 'student')\n",
                "endpoint_name = f'{user}-pa4-analyst-endpoint'\n",
                "\n",
                "c = DocumentAnalystClient(endpoint_name)\n",
                "assert c.health_check() is True\n",
                "print(c.ask('What was the net income in 2023?'))\n"
            ]
        elif "# ask_streaming demo" in src:
            cell["source"] = [
                "# ask_streaming demo\n",
                "for chunk in c.ask_streaming('Summarize FY2023 revenue.'):\n",
                "    print(chunk, end='', flush=True)\n",
                "print()\n"
            ]
        elif "# Simulate timeout (timeout=0.001) and endpoint-unavailable retry behavior" in src:
            cell["source"] = [
                "# Simulate timeout (timeout=0.001)\n",
                "c_timeout = DocumentAnalystClient(endpoint_name, timeout=0.001)\n",
                "try:\n",
                "    c_timeout.ask('What is the company name?')\n",
                "except TimeoutError as e:\n",
                "    print(f'Caught TimeoutError successfully: {e}')\n",
                "\n",
                "# Simulate endpoint-unavailable retry behavior (using a non-existent endpoint)\n",
                "c_unavailable = DocumentAnalystClient('non-existent-endpoint-12345', max_retries=1)\n",
                "try:\n",
                "    c_unavailable.ask('Will fail')\n",
                "except Exception as e:\n",
                "    print(f'Caught Exception for unavailable endpoint successfully: {e}')\n"
            ]

with open("pa4.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
