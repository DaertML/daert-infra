#!/bin/bash

set -e

echo ""
echo "=============================================="
echo "    AutoML Experiment Manager - CLI"
echo "=============================================="
echo ""

export PS1="\[\033[1;32m\]automl\[\033[0m\]@\[\033[1;34m\]container\[\033[0m\]:\w\$ "

if [ -t 0 ]; then
    exec /bin/bash --login
else
    exec /bin/bash "$@"
fi
