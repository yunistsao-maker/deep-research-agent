"""One-shot setup for the experiment (idempotent-ish; re-running reuses existing resources by name).

Creates:
  1. S3 bucket for run results
  2. Gateway service role  (bedrock-agentcore:InvokeWebSearch on the AWS-owned web-search tool)
  3. AgentCore Gateway     (MCP, inbound auth AWS_IAM -> the runtime role signs with SigV4)
  4. Web Search target     (connectorId "web-search")
  5. Runtime execution role (Bedrock models, gateway, browser, S3, logs/traces, ECR)
  6. AgentCore Runtime      (container image you pushed; PUBLIC network mode so S2/S3 can fetch)

Usage:
  python infra/setup.py --region us-west-2 --image <acct>.dkr.ecr.us-west-2.amazonaws.com/deep-research-agent:latest
Writes infra/outputs.json (gateway URL, runtime ARN, bucket) for eval/run_matrix.py.

Web Search regions (docs, 2026-10): us-east-1, us-east-2, us-west-2, eu-west-1, ap-southeast-2, ap-northeast-1.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import boto3
from botocore.exceptions import ClientError

NAME = "deep-research-eval"


def trust(service: str, account: str) -> str:
    return json.dumps({"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]})


def ensure_role(iam, name: str, trust_doc: str, policy: dict) -> str:
    try:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=trust_doc)["Role"]["Arn"]
        time.sleep(10)  # IAM propagation
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    iam.put_role_policy(RoleName=name, PolicyName=f"{name}-policy", PolicyDocument=json.dumps(policy))
    return arn


def wait(fn, key: str, ok=("READY",), bad=("FAILED", "CREATE_FAILED", "UPDATE_FAILED"), timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = fn()[key]
        if st in ok:
            return
        if st in bad:
            raise RuntimeError(f"resource failed: {fn()}")
        time.sleep(5)
    raise TimeoutError(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--image", required=True, help="ECR image URI (linux/arm64)")
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--model-id", default="global.anthropic.claude-sonnet-5-5")
    args = ap.parse_args()

    sess = boto3.Session(region_name=args.region)
    account = sess.client("sts").get_caller_identity()["Account"]
    region = args.region
    iam, s3 = sess.client("iam"), sess.client("s3")
    ctl = sess.client("bedrock-agentcore-control")
    bucket = args.bucket or f"{NAME}-{account}-{region}"

    # 1) bucket
    try:
        kw = {} if region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": region}}
        s3.create_bucket(Bucket=bucket, **kw)
        s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou",):
            raise
    print("bucket:", bucket)

    # 2) gateway service role
    gw_role = ensure_role(iam, f"{NAME}-gateway-role", trust("bedrock-agentcore.amazonaws.com", account), {
        "Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": "bedrock-agentcore:InvokeWebSearch",
             "Resource": f"arn:aws:bedrock-agentcore:{region}:aws:tool/web-search.v1"},
            {"Effect": "Allow", "Action": "bedrock-agentcore:InvokeGateway",
             "Resource": f"arn:aws:bedrock-agentcore:{region}:{account}:gateway/*"}]})

    # 3) gateway (AWS_IAM inbound)
    gw = next((g for g in ctl.list_gateways().get("items", []) if g["name"] == f"{NAME}-gw"), None)
    if gw:
        gw = ctl.get_gateway(gatewayIdentifier=gw["gatewayId"])
    else:
        gw = ctl.create_gateway(name=f"{NAME}-gw", roleArn=gw_role, protocolType="MCP", authorizerType="AWS_IAM",
                                description="Web Search connector for deep research tool-setting experiment")
    gw_id = gw["gatewayId"]
    wait(lambda: ctl.get_gateway(gatewayIdentifier=gw_id), "status")
    gw = ctl.get_gateway(gatewayIdentifier=gw_id)
    print("gateway:", gw["gatewayUrl"])

    # 4) web search target
    targets = ctl.list_gateway_targets(gatewayIdentifier=gw_id).get("items", [])
    tgt = next((t for t in targets if t["name"] == "web-search"), None)
    if not tgt:
        tgt = ctl.create_gateway_target(
            gatewayIdentifier=gw_id, name="web-search",
            targetConfiguration={"mcp": {"connector": {"source": {"connectorId": "web-search"},
                                                       "configurations": [{"name": "WebSearch", "parameterValues": {}}]}}},
            credentialProviderConfigurations=[{"credentialProviderType": "GATEWAY_IAM_ROLE"}])
    wait(lambda: ctl.get_gateway_target(gatewayIdentifier=gw_id, targetId=tgt["targetId"]), "status")
    print("target: web-search  (tool name: web-search___WebSearch)")

    # 5) runtime role
    rt_role = ensure_role(iam, f"{NAME}-runtime-role", trust("bedrock-agentcore.amazonaws.com", account), {
        "Version": "2012-10-17", "Statement": [
            {"Sid": "Models", "Effect": "Allow",
             "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
             "Resource": ["arn:aws:bedrock:*::foundation-model/*", f"arn:aws:bedrock:*:{account}:inference-profile/*"]},
            {"Sid": "Gateway", "Effect": "Allow", "Action": "bedrock-agentcore:InvokeGateway", "Resource": gw["gatewayArn"]},
            {"Sid": "Browser", "Effect": "Allow",
             "Action": ["bedrock-agentcore:StartBrowserSession", "bedrock-agentcore:StopBrowserSession",
                        "bedrock-agentcore:GetBrowserSession", "bedrock-agentcore:ConnectBrowserAutomationStream",
                        "bedrock-agentcore:ConnectBrowserLiveViewStream", "bedrock-agentcore:ListBrowserSessions"],
             "Resource": "*"},
            {"Sid": "Results", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"],
             "Resource": f"arn:aws:s3:::{bucket}/*"},
            {"Sid": "Ecr", "Effect": "Allow",
             "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:GetAuthorizationToken"], "Resource": "*"},
            {"Sid": "Obs", "Effect": "Allow",
             "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams",
                        "logs:DescribeLogGroups", "xray:PutTraceSegments", "xray:PutTelemetryRecords",
                        "cloudwatch:PutMetricData"], "Resource": "*"},
            {"Sid": "Identity", "Effect": "Allow",
             "Action": ["bedrock-agentcore:GetWorkloadAccessToken", "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                        "bedrock-agentcore:GetWorkloadAccessTokenForUserId"], "Resource": "*"}]})

    # 6) runtime
    env = {"AWS_REGION": region, "GATEWAY_URL": gw["gatewayUrl"], "WEB_SEARCH_TOOL_NAME": "web-search___WebSearch",
           "MODEL_ID": args.model_id, "RESULTS_BUCKET": bucket, "RESULTS_PREFIX": "deep-research-runs",
           "RESPECT_ROBOTS": "true"}
    spec = dict(agentRuntimeArtifact={"containerConfiguration": {"containerUri": args.image}},
                roleArn=rt_role, networkConfiguration={"networkMode": "PUBLIC"},
                protocolConfiguration={"serverProtocol": "HTTP"}, environmentVariables=env,
                lifecycleConfiguration={"idleRuntimeSessionTimeout": 900, "maxLifetime": 28800})
    existing = next((r for r in ctl.list_agent_runtimes().get("agentRuntimes", [])
                     if r["agentRuntimeName"] == NAME.replace("-", "_")), None)
    if existing:
        rt = ctl.update_agent_runtime(agentRuntimeId=existing["agentRuntimeId"], **spec)
    else:
        rt = ctl.create_agent_runtime(agentRuntimeName=NAME.replace("-", "_"), **spec)
    wait(lambda: ctl.get_agent_runtime(agentRuntimeId=rt["agentRuntimeId"]), "status")
    print("runtime:", rt["agentRuntimeArn"])

    out = {"region": region, "bucket": bucket, "results_prefix": "deep-research-runs",
           "gateway_id": gw_id, "gateway_url": gw["gatewayUrl"], "runtime_arn": rt["agentRuntimeArn"],
           "model_id": args.model_id}
    path = os.path.join(os.path.dirname(__file__), "outputs.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
