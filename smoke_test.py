import os
from pathlib import Path
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["RF_API_KEY"].strip(),
)

img = "data/batch_images/00001.jpg"

print("WS:", os.environ["RF_WORKSPACE"])
print("WF:", os.environ["RF_WORKFLOW_ID"])

result = client.run_workflow(
    workspace_name=os.environ["RF_WORKSPACE"].strip(),
    workflow_id=os.environ["RF_WORKFLOW_ID"].strip(),
    images={"image": img},
    use_cache=False
)

print("OK")
print(result)