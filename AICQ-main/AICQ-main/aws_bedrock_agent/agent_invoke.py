import os
import json
import boto3

"""
Invoke the Bedrock Agent alias with a simple moisture-text classification instruction.

Env vars:
- AWS_REGION
- AGENT_ID
- AGENT_ALIAS_ID
- TEST_TEXT (optional)
"""


def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def main():
    region = get_env('AWS_REGION', required=True)
    agent_id = get_env('AGENT_ID', required=True)
    alias_id = get_env('AGENT_ALIAS_ID', required=True)
    text = get_env('TEST_TEXT', 'slightly damp subgrade, run option1 pipeline')

    client = boto3.client('bedrock-agent-runtime', region_name=region)
    resp = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId='demo-session',
        inputText=text,
        enableTrace=True,
    )
    # Stream response chunks
    for event in resp.get('completion', []):
        print(event)


if __name__ == '__main__':
    main()
