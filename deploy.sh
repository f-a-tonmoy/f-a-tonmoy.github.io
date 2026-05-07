#!/usr/bin/env bash
# Deploy the portfolio to GitHub Pages.
#
# Usage:
#   ./deploy.sh                     # commits with a default message
#   ./deploy.sh "your message"      # commits with a custom message
#
# The repo is already linked to https://github.com/f-a-tonmoy/f-a-tonmoy.github.io
# Pushing to `main` triggers GitHub Pages to redeploy in ~30-60 seconds.

set -e  # exit immediately on any error

cd "$(dirname "$0")"

# Refuse to run if there are no changes
if [[ -z "$(git status --porcelain)" ]]; then
    echo "Nothing to commit — working tree is clean."
    exit 0
fi

MESSAGE="${1:-Update portfolio site}"

echo "==> Staging all changes..."
git add -A

echo "==> Committing..."
git commit -m "$MESSAGE"

echo "==> Pushing to origin/main..."
git push origin main

echo
echo "Done. Live in ~30-60s at: https://f-a-tonmoy.github.io"
