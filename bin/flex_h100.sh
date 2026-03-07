#!/bin/sh
gcloud compute instances create h100-flex \
    --project=ml-ai-420606 \
    --zone=us-central1-a \
    --machine-type=a3-highgpu-1g \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --metadata=enable-osconfig=TRUE,install-nvidia-driver=True \
    --metadata-from-file=startup-script=bin/train.sh \
    --maintenance-policy=TERMINATE \
    --provisioning-model=FLEX_START \
    --instance-termination-action=DELETE \
    --max-run-duration=172800s \
    --service-account=996102297610-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --accelerator=count=1,type=nvidia-h100-80gb \
    --create-disk=auto-delete=yes,boot=yes,device-name=h100-flex,image=projects/ml-images/global/images/c0-deeplearning-common-cu124-v20250325-debian-11-py310-conda,mode=rw,size=200,type=pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --reservation-affinity=none
