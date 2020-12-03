#!/usr/bin/env bash
#  ./git_push.sh "message content"
message=$1
branch=$2

find . -name .DS_Store -print0 | xargs -0 git rm -rf --ignore-unmatch

find . -name __pycache__ -print0 | xargs -0 git rm -rf --ignore-unmatch

git rm -rf --cached .

git add .

git commit -m "${message}"

git push origin ${branch}

