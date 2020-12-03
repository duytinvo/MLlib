#!/usr/bin/env bash

# vmname=nlp-v100-04
vmname=$1

gcloud compute instances start --project mystic-sound-143520 --zone us-central1-f ${vmname}

gcloud compute instances stop --project mystic-sound-143520 --zone us-central1-f ${vmname}

gcloud compute --project "mystic-sound-143520" ssh --zone "us-central1-f" "chatadevops@nlp-v100-04"

gcloud compute scp --project "mystic-sound-143520" --zone "us-central1-f" --recurse "chatadevops@nlp-v100-04:/media/data/review_response/trained_model/checkpoint_by-epoch_*/*" /media/data/review_response/rouge_response_model/

