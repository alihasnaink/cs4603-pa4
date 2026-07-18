import json

with open('pa4.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    source = "".join(cell.get('source', []))
    
    if "# TODO(2.2): log + register" in source:
        cell['source'] = [
            "# TODO(2.2): log + register the model version in Unity Catalog\n",
            "import os, sys\n",
            "sys.path.insert(0, os.path.abspath('.'))\n",
            "from deployment.deploy import log_and_register\n",
            "uc_name, version = log_and_register()\n"
        ]
    elif "# TODO(2.3): create/update the serving endpoint" in source:
        cell['source'] = [
            "# TODO(2.3): create/update the serving endpoint; wait for READY; print the URL\n",
            "from deployment.deploy import create_or_update_endpoint\n",
            "endpoint_url = create_or_update_endpoint(uc_name, version)\n"
        ]
    elif "# curl the endpoint and show the raw response" in source:
        cell['source'] = [
            "# curl the endpoint and show the raw response\n",
            "import os\n",
            "host = os.environ.get('DATABRICKS_HOST', '')\n",
            "token = os.environ.get('DATABRICKS_TOKEN', '')\n",
            "user = os.environ.get('USER', 'student')\n",
            "endpoint_name = f'{user}-pa4-analyst-endpoint'\n",
            "url = f'{host}/serving-endpoints/{endpoint_name}/invocations'\n",
            "print(f'Endpoint URL: {url}')\n",
            "\n",
            "!curl -s -X POST -H 'Authorization: Bearer {token}' \\\n",
            "  -H 'Content-Type: application/json' \\\n",
            "  -d '{\"messages\": [{\"role\": \"user\", \"content\": \"What was the net income in 2023?\"}]}' \\\n",
            "  {url}\n"
        ]
    elif "# Response shape depends on how you logged the model" in source:
        cell['source'] = [
            "# Response shape depends on how you logged the model (see README Task 2.4 / GUIDE §7).\n",
            "import requests\n",
            "import time\n",
            "\n",
            "def query_endpoint(question):\n",
            "    start = time.time()\n",
            "    resp = requests.post(url, headers={'Authorization': f'Bearer {token}'}, json={'messages': [{'role': 'user', 'content': question}]})\n",
            "    latency = time.time() - start\n",
            "    data = resp.json()\n",
            "    try:\n",
            "        answer = data[0]['messages'][-1]['content']\n",
            "    except Exception:\n",
            "        answer = data\n",
            "    return answer, latency\n",
            "\n",
            "ans, lat = query_endpoint('What was the net income in 2023?')\n",
            "print(f'Answer: {ans}\\nLatency: {lat:.2f}s')\n",
            "\n",
            "print('-'*40)\n",
            "q1 = 'What was the net income in 2023?'\n",
            "q2 = 'What is 15% of 2.4 billion?'\n",
            "q3 = 'What was the revenue in 2023, and what would a 10% increase look like?'\n",
            "for q in [q1, q2, q3]:\n",
            "    a, l = query_endpoint(q)\n",
            "    print(f'Q: {q}\\nA: {a}\\n[Latency: {l:.2f}s]\\n')\n"
        ]

with open('pa4.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Updated pa4.ipynb successfully.")
