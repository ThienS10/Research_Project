import os
import time
import json
import boto3

"""
Provision an Amazon Bedrock Agent with a Lambda action group that calls the
award pipeline API (deployed via SAM) or a separate Lambda function.

Config via environment variables:
- AWS_REGION: e.g., us-east-1
- AGENT_NAME: logical display name for the agent (default: MoistureAgent)
- AGENT_INSTRUCTION: system prompt/instruction (optional)
- AGENT_FOUNDATION_MODEL: e.g., anthropic.claude-3-5-sonnet-20240620-v1:0
- LAMBDA_ARN: ARN of the Lambda the agent will call as a tool
- AGENT_ALIAS_NAME: alias name to create/update (default: prod)

Outputs (printed): agent_id, agent_alias_id
"""

def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def find_agent_id_by_name(client, name: str) -> str | None:
    paginator = client.get_paginator('list_agents')
    for page in paginator.paginate():
        for a in page.get('agentSummaries', []):
            if a.get('agentName') == name:
                return a.get('agentId')
    return None


def ensure_agent(client, name: str, model_id: str, instruction: str | None) -> str:
    agent_id = find_agent_id_by_name(client, name)
    if agent_id:
        # Update instruction or model if provided
        try:
            client.update_agent(
                agentId=agent_id,
                agentName=name,
                foundationModel=model_id,
                instruction=instruction or "",
                idleSessionTTLInSeconds=600,
            )
        except Exception:
            pass
        return agent_id
    resp = client.create_agent(
        agentName=name,
        foundationModel=model_id,
        instruction=instruction or "Classify moisture text and call the tool to process datasets.",
        idleSessionTTLInSeconds=600,
    )
    return resp['agent']['agentId']


def ensure_action_group(client, agent_id: str, lambda_arn: str, name: str = "RunPipeline") -> str:
    # Try to create; if exists, update
    try:
        resp = client.create_agent_action_group(
            agentId=agent_id,
            actionGroupName=name,
            actionGroupExecutor={"lambda": {"lambdaArn": lambda_arn}},
            description="Invoke pipeline Lambda",
            apiSchema={"payload": json.dumps({"openapi": "3.0.0", "info": {"title": "Pipeline", "version": "1.0.0"}})},
        )
        return resp['agentActionGroup']['actionGroupId']
    except Exception:
        # list and find existing, then update
        groups = client.list_agent_action_groups(agentId=agent_id).get('agentActionGroupSummaries', [])
        ag_id = None
        for g in groups:
            if g.get('actionGroupName') == name:
                ag_id = g.get('actionGroupId')
                break
        if not ag_id:
            raise
        try:
            client.update_agent_action_group(
                agentId=agent_id,
                actionGroupId=ag_id,
                actionGroupName=name,
                actionGroupExecutor={"lambda": {"lambdaArn": lambda_arn}},
                description="Invoke pipeline Lambda",
            )
        except Exception:
            pass
        return ag_id


def ensure_alias(client, agent_id: str, alias_name: str) -> str:
    # Try create alias, else find/update
    try:
        resp = client.create_agent_alias(agentId=agent_id, agentAliasName=alias_name)
        return resp['agentAlias']['agentAliasId']
    except Exception:
        aliases = client.list_agent_aliases(agentId=agent_id).get('agentAliasSummaries', [])
        for a in aliases:
            if a.get('agentAliasName') == alias_name:
                return a.get('agentAliasId')
        raise


def prepare_and_deploy(client, agent_id: str):
    try:
        client.prepare_agent(agentId=agent_id)
    except Exception:
        pass
    # Simple wait loop for prepared state
    for _ in range(30):
        try:
            desc = client.get_agent(agentId=agent_id)
            if desc.get('agent', {}).get('agentStatus') in ("PREPARED", "READY"):
                break
        except Exception:
            pass
        time.sleep(5)


def main():
    region = get_env('AWS_REGION', required=True)
    agent_name = get_env('AGENT_NAME', 'MoistureAgent')
    model_id = get_env('AGENT_FOUNDATION_MODEL', required=True)
    lambda_arn = get_env('LAMBDA_ARN', required=True)
    instruction = get_env('AGENT_INSTRUCTION', None)
    alias_name = get_env('AGENT_ALIAS_NAME', 'prod')

    bedrock_agents = boto3.client('bedrock-agent', region_name=region)

    agent_id = ensure_agent(bedrock_agents, agent_name, model_id, instruction)
    action_group_id = ensure_action_group(bedrock_agents, agent_id, lambda_arn)
    prepare_and_deploy(bedrock_agents, agent_id)
    alias_id = ensure_alias(bedrock_agents, agent_id, alias_name)

    print(json.dumps({
        "agent_id": agent_id,
        "action_group_id": action_group_id,
        "alias_id": alias_id,
    }, indent=2))


if __name__ == '__main__':
    main()
