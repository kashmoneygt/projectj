#!/usr/bin/env bash
# deploy jym to Azure Container Apps, creating any missing resources; safe to rerun
#   deploy/azure.sh        build the committed code and roll it out
#   deploy/azure.sh down   delete the resource group and everything in it

set -euo pipefail
cd "$(dirname "$0")/.."

LOCATION=eastus2
GROUP=jym-rg
REGISTRY=jymcr
STORAGE=jymst
SHARE=jym-data
ENVIRONMENT=jym-env
IDENTITY=jym-pull  # pulls images from the registry
APP=jym
CPU=2
MEMORY=4Gi  # Container Apps gives each vCPU 2 GiB
TAG=$(git rev-parse --short=12 HEAD)

say() { printf '\n== %s\n' "$*"; }

if [[ "${1:-}" == "down" ]]; then
  say "Deleting resource group ${GROUP}"
  az group delete --name "$GROUP" --yes
  exit 0
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
az extension add --name containerapp --upgrade --only-show-errors

say "Resource group ${GROUP} and container registry ${REGISTRY}"
if ! az group show --name "$GROUP" --output none 2>/dev/null; then
  az group create --name "$GROUP" --location "$LOCATION" --output none
fi
if ! az acr show --name "$REGISTRY" --resource-group "$GROUP" --output none 2>/dev/null; then
  az acr create --name "$REGISTRY" --resource-group "$GROUP" --location "$LOCATION" --sku Basic \
    --admin-enabled false --output none
fi
LOGIN_SERVER=$(az acr show --name "$REGISTRY" --query loginServer -o tsv)
IMAGE="${LOGIN_SERVER}/jym:${TAG}"

say "Building jym:${TAG} from the committed code"
if az acr repository show-tags --name "$REGISTRY" --repository jym -o tsv 2>/dev/null | grep -qx "$TAG"; then
  echo "the image is already built"
else
  mkdir "$WORK/context"
  git archive HEAD | tar -x -C "$WORK/context"  # from jym/, packs only this folder as committed
  az acr build --registry "$REGISTRY" --image "jym:${TAG}" --file deploy/Dockerfile "$WORK/context" --output none
fi
# keep the last five images (fits the Basic registry's storage)
az acr run --registry "$REGISTRY" --cmd "acr purge --filter 'jym:.*' --keep 5 --ago 0d --untagged" /dev/null \
  --output none >/dev/null || echo "could not prune old images" >&2

say "Storage account ${STORAGE} and its file share ${SHARE}"
if ! az storage account show --name "$STORAGE" --resource-group "$GROUP" --output none 2>/dev/null; then
  az storage account create --name "$STORAGE" --resource-group "$GROUP" --location "$LOCATION" --sku Standard_LRS \
    --kind StorageV2 --min-tls-version TLS1_2 --allow-blob-public-access false --output none
fi
if ! az storage share-rm show --storage-account "$STORAGE" --resource-group "$GROUP" --name "$SHARE" \
  --output none 2>/dev/null; then
  az storage share-rm create --storage-account "$STORAGE" --resource-group "$GROUP" --name "$SHARE" --quota 5 \
    --output none
fi

say "Container Apps environment ${ENVIRONMENT}, the share mounted in it"
if ! az containerapp env show --name "$ENVIRONMENT" --resource-group "$GROUP" --output none 2>/dev/null; then
  # no Log Analytics; use az containerapp logs show
  az containerapp env create --name "$ENVIRONMENT" --resource-group "$GROUP" --location "$LOCATION" \
    --logs-destination none --enable-workload-profiles false --output none
fi
KEY=$(az storage account keys list --account-name "$STORAGE" --resource-group "$GROUP" --query "[0].value" -o tsv)
az containerapp env storage set --name "$ENVIRONMENT" --resource-group "$GROUP" --storage-name "$SHARE" \
  --azure-file-account-name "$STORAGE" --azure-file-account-key "$KEY" --azure-file-share-name "$SHARE" \
  --access-mode ReadWrite --output none
ENVIRONMENT_ID=$(az containerapp env show --name "$ENVIRONMENT" --resource-group "$GROUP" --query id -o tsv)
HOST="${APP}.$(az containerapp env show --name "$ENVIRONMENT" --resource-group "$GROUP" \
  --query properties.defaultDomain -o tsv)"

say "Identity ${IDENTITY}, allowed to pull from ${REGISTRY}"
if ! az identity show --name "$IDENTITY" --resource-group "$GROUP" --output none 2>/dev/null; then
  az identity create --name "$IDENTITY" --resource-group "$GROUP" --location "$LOCATION" --output none
fi
IDENTITY_ID=$(az identity show --name "$IDENTITY" --resource-group "$GROUP" --query id -o tsv)
PRINCIPAL=$(az identity show --name "$IDENTITY" --resource-group "$GROUP" --query principalId -o tsv)
REGISTRY_ID=$(az acr show --name "$REGISTRY" --query id -o tsv)
if [[ -z $(az role assignment list --scope "$REGISTRY_ID" --role AcrPull \
  --query "[?principalId=='${PRINCIPAL}'].id" -o tsv) ]]; then
  az role assignment create --assignee-object-id "$PRINCIPAL" --assignee-principal-type ServicePrincipal \
    --scope "$REGISTRY_ID" --role AcrPull --output none
fi

say "Container app ${APP}: jym:${TAG}"
cat >"$WORK/app.yaml" <<YAML
location: ${LOCATION}
identity:
  type: UserAssigned
  userAssignedIdentities:
    ${IDENTITY_ID}: {}
properties:
  managedEnvironmentId: ${ENVIRONMENT_ID}
  configuration:
    activeRevisionsMode: Single
    ingress:
      external: true
      targetPort: 8080
      allowInsecure: false
    registries:
      - server: ${LOGIN_SERVER}
        identity: ${IDENTITY_ID}
  template:
    containers:
      - name: jym
        image: ${IMAGE}
        resources:
          cpu: ${CPU}
          memory: ${MEMORY}
        env:
          - name: JYM_HOSTS
            value: ${HOST}
          - name: LLAMA_THREADS
            value: "${CPU}"
        probes:
          # the old revision serves visitors until the new one has loaded its models
          - type: Readiness
            httpGet:
              path: /api/health?ready=1
              port: 8080
            periodSeconds: 5
            failureThreshold: 3
        volumeMounts:
          - volumeName: data
            mountPath: /home/data
    volumes:
      - name: data
        storageType: AzureFile
        storageName: ${SHARE}
        mountOptions: uid=10001,gid=10001,dir_mode=0750,file_mode=0640
    # one always-on replica, so models are loaded before anyone visits
    scale:
      minReplicas: 1
      maxReplicas: 1
YAML
if az containerapp show --name "$APP" --resource-group "$GROUP" --output none 2>/dev/null; then
  az containerapp update --name "$APP" --resource-group "$GROUP" --yaml "$WORK/app.yaml" --output none
else
  az containerapp create --name "$APP" --resource-group "$GROUP" --yaml "$WORK/app.yaml" --output none
fi

say "Waiting for jym:${TAG} to take over at https://${HOST}/"
for _ in $(seq 1 60); do
  REVISIONS=$(az containerapp show --name "$APP" --resource-group "$GROUP" \
    --query "[properties.latestRevisionName, properties.latestReadyRevisionName]" -o tsv | paste -sd' ')
  read -r LATEST READY <<<"$REVISIONS"
  if [[ -n "${LATEST:-}" && "$LATEST" == "${READY:-}" ]] \
    && curl -fsS "https://${HOST}/api/health?ready=1" >/dev/null 2>&1; then
    echo "jym is up: https://${HOST}/"
    exit 0
  fi
  sleep 10
done
echo "jym:${TAG} was not ready in 10 minutes: az containerapp logs show --name ${APP} --resource-group ${GROUP}" >&2
exit 1
