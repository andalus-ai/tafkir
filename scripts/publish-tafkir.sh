#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Publishing Tafkir to GitHub Packages..."

if [[ -z "${GITHUB_REPOSITORY_OWNER:-}" ]]; then
  if git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    remote_url="$(git -C "$PROJECT_ROOT" config --get remote.origin.url || true)"
    if [[ "$remote_url" =~ (github\.com|github-ngoding)[:/]+([^/]+)/[^/]+(\.git)?$ ]]; then
      GITHUB_REPOSITORY_OWNER="${BASH_REMATCH[2]}"
    elif [[ "$remote_url" =~ git@[^:]+:([^/]+)/[^/]+(\.git)?$ ]]; then
      GITHUB_REPOSITORY_OWNER="${BASH_REMATCH[1]}"
    elif [[ "$remote_url" =~ https?://[^/]+/([^/]+)/[^/]+(\.git)?$ ]]; then
      GITHUB_REPOSITORY_OWNER="${BASH_REMATCH[1]}"
    fi
  fi
fi

GITHUB_REPOSITORY_OWNER="${GITHUB_REPOSITORY_OWNER:-${GITHUB_ACTOR:-bhangun}}"
export GITHUB_REPOSITORY_OWNER

if [[ -z "${GITHUB_ACTOR:-}" || -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Error: GITHUB_ACTOR and GITHUB_TOKEN environment variables must be set."
  echo "Use a PAT with 'read:packages' and 'write:packages' scopes."
  echo "Example: export GITHUB_ACTOR=bhangun; export GITHUB_TOKEN=..."
  exit 1
fi

cd "$PROJECT_ROOT"

echo "Repository owner: ${GITHUB_REPOSITORY_OWNER}"
echo "Publishing to: https://maven.pkg.github.com/${GITHUB_REPOSITORY_OWNER}"

# This publishes to the GitHub Packages repository configured in build.gradle.kts.
# GitHub Packages Maven endpoints are owner-scoped, not repository-scoped.
./gradlew --no-daemon publishAllPublicationsToGitHubPackagesRepository

echo "✅ Successfully published to GitHub Packages."
