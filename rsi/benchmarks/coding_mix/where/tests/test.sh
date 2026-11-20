#!/bin/bash
# grade a where answer with whatever python 3.8+ the image has
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
for py in /opt/harbor-grader/python3 python3 python /opt/conda/bin/python3; do
  py=$(command -v "$py" 2> /dev/null) && "$py" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2> /dev/null && break
  py=""
done
if [ -z "$py" ]; then
  echo "no python 3.8 or newer to grade with"
  exit 0
fi
"$py" -I -S /tests/grade.py
