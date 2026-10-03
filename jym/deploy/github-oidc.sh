#!/usr/bin/env bash
# let GitHub Actions run deploy/azure.sh with OpenID Connect (no stored Azure secrets)
# run once locally, as a user who can create app registrations and role assignments; safe to rerun
set -euo pipefail

GROUP=jym-rg
LOCATION=eastus2
DEPLOY_APP=jym-deploy
ISSUER=https://token.actions.githubusercontent.com
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
SUBJECT="repo:${REPO}:ref:refs/heads/main"
SUBSCRIPTION=$(az account show --query id -o tsv)
TENANT=$(az account show --query tenantId -o tsv)

if ! az group show --name "$GROUP" --output none 2>/dev/null; then
  az group create --name "$GROUP" --location "$LOCATION" --output none
fi

APP_ID=$(az ad app list --display-name "$DEPLOY_APP" --query "[0].appId" -o tsv)
if [[ -z "$APP_ID" ]]; then
  APP_ID=$(az ad app create --display-name "$DEPLOY_APP" --query appId -o tsv)
fi
if [[ -z $(az ad sp list --filter "appId eq '${APP_ID}'" --query "[0].id" -o tsv) ]]; then
  az ad sp create --id "$APP_ID" --output none
fi
# delete other credentials so no other repo, branch or event can deploy
for credential in $(az ad app federated-credential list --id "$APP_ID" \
  --query "[?issuer!='${ISSUER}' || subject!='${SUBJECT}'].id" -o tsv); do
  az ad app federated-credential delete --id "$APP_ID" --federated-credential-id "$credential"
done
if [[ -z $(az ad app federated-credential list --id "$APP_ID" --query "[?subject=='${SUBJECT}'].id" -o tsv) ]]; then
  az ad app federated-credential create --id "$APP_ID" --parameters "{
    \"name\": \"github-main\",
    \"issuer\": \"${ISSUER}\",
    \"subject\": \"${SUBJECT}\",
    \"audiences\": [\"api://AzureADTokenExchange\"]
  }" --output none
fi
SCOPE=$(az group show --name "$GROUP" --query id -o tsv)
ASSIGNED=$(az role assignment list --assignee "$APP_ID" --scope "$SCOPE" --role Contributor --query "[].id" -o tsv)
if [[ -z "$ASSIGNED" ]]; then
  az role assignment create --assignee "$APP_ID" --scope "$SCOPE" --role Contributor --output none
fi

gh secret set AZURE_CLIENT_ID --repo "$REPO" --body "$APP_ID"
gh secret set AZURE_TENANT_ID --repo "$REPO" --body "$TENANT"
gh secret set AZURE_SUBSCRIPTION_ID --repo "$REPO" --body "$SUBSCRIPTION"
echo "GitHub Actions in ${REPO} can now deploy from main to ${GROUP} as ${DEPLOY_APP} (${APP_ID})."
