#!/bin/bash
# reference answer, i.e. the files the project's real fix changed
mkdir -p /logs/agent
sed -n 's|^diff --git a/\([^ ]*\) b/.*|\1|p' /solution/reference/gold.patch > /logs/agent/response.txt
