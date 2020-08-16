  
#!/bin/bash

set -e

log() {
  echo "[$(date +%F_%H:%M:%S)] - " $@
}

SECRET_FILE=./secret.yml

# AWS ECR Login
log "Executing AWS ECR Login"
aws ecr get-login-password --region eu-west-3 | docker login --username AWS --password-stdin 505799072829.dkr.ecr.eu-west-3.amazonaws.com
log "New AWS ECR token generated."

# Replace secret value
dockerconfigjson_base64=`echo -n "$(cat /root/.docker/config.json)" | base64 | tr -d '\n'`
echo -n \
"---
kind: Secret
apiVersion: v1
metadata:
  name: aws-ecr
data:
  .dockerconfigjson: $dockerconfigjson_base64
type: kubernetes.io/dockerconfigjson
" \
> $SECRET_FILE

log "Login as system:admin ... " && oc login -u system:admin || log "Unable to connect to Openshift cluster"
oc version

log "Creating/Updating secrets and configure service account on Openshift"

PROJECTS=$(oc get projects --no-headers | awk '{print $1}')

SAs=(default deployer builder)

for project in $PROJECTS; do
  log "Project: $project"
  found=$(oc get secrets -n $project | grep aws-ecr | wc -l)
  if [[ $found -eq 0 ]]; then
    oc create -f ${SECRET_FILE} -n $project
    log "Created secret aws-ecr"
    for sa in ${SAs[@]};do
      oc secrets add serviceaccount/$sa secrets/aws-ecr --for=pull -n $project
      log "Added secret to serviceaccount $sa"
    done
  else
    oc replace -f ${SECRET_FILE} -n $project
    log "Updated secret aws-ecr"
  fi
done

PROJECTS=$(oc get projects --no-headers | awk '{print $1}' | grep tfmiot)

SAs=(backend worker filebeat)

for project in $PROJECTS; do
  log "Project: $project"
  found=$(oc get secrets -n $project | grep aws-ecr | wc -l)
  if [[ $found -eq 0 ]]; then
    oc create -f ${SECRET_FILE} -n $project
    log "Created secret aws-ecr"
    for sa in ${SAs[@]};do
      oc secrets add serviceaccount/$sa secrets/aws-ecr --for=pull -n $project
      log "Added secret to serviceaccount $sa"
    done
  else
    oc replace -f ${SECRET_FILE} -n $project
    log "Updated secret aws-ecr"
  fi
done
