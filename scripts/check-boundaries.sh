#!/usr/bin/env bash
# =============================================================================
# Tafkir — Architectural Boundary Enforcement Script
#
# Enforces architectural rules:
#   Check 1: No phantom/ghost stub directories (stubs/ or backend/ or core/)
#   Check 2: No modules published under foreign groups (alkhawarizm or gollek)
#   Check 3: No pseudo-project references to Alkhawarizm (:core:alkhawarizm-* or :backend:*)
#   Check 4: All modules declare canonical group (tech.kayys.tafkir)
#
# Exit code: 0 if all checks pass, 1 if any check fails.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAFKIR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TAFKIR_ROOT}"

# ---------------------------------------------------------------------------
# Color helpers (tput when available, plain fallback)
# ---------------------------------------------------------------------------
if command -v tput &>/dev/null && tput setaf 1 &>/dev/null; then
  RED="$(tput setaf 1)"
  GREEN="$(tput setaf 2)"
  YELLOW="$(tput setaf 3)"
  BOLD="$(tput bold)"
  RESET="$(tput sgr0)"
else
  RED="" GREEN="" YELLOW="" BOLD="" RESET=""
fi

TOTAL_CHECKS=0
FAILED_CHECKS=0
FAILURES=()

header() {
  echo
  echo "${BOLD}─────────────────────────────────────────────────────────────${RESET}"
  echo "${BOLD}  $*${RESET}"
  echo "${BOLD}─────────────────────────────────────────────────────────────${RESET}"
}

pass() {
  echo "  ${GREEN}✅  $*${RESET}"
}

fail() {
  echo "  ${RED}❌  $*${RESET}"
  FAILURES+=("$*")
  FAILED_CHECKS=$((FAILED_CHECKS + 1))
}

warn() {
  echo "  ${YELLOW}⚠️   $*${RESET}"
}

run_check() {
  local description="$1"
  shift

  TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

  local output
  output=$("$@" 2>&1 || true)

  if [[ -z "${output}" ]]; then
    pass "${description}"
  else
    fail "${description}"
    while IFS= read -r line; do
      echo "    ${RED}${line}${RESET}"
    done <<< "${output}"
  fi
}

run_warn_check() {
  local description="$1"
  shift

  local output
  output=$("$@" 2>&1 || true)

  if [[ -z "${output}" ]]; then
    pass "${description}"
  else
    warn "${description}"
    while IFS= read -r line; do
      echo "    ${YELLOW}${line}${RESET}"
    done <<< "${output}"
  fi
}

# =============================================================================
# Banner
# =============================================================================
echo
echo "${BOLD}=============================================================${RESET}"
echo "${BOLD}  Tafkir — Architectural Boundary Checks${RESET}"
echo "${BOLD}  Root: ${TAFKIR_ROOT}${RESET}"
echo "${BOLD}=============================================================${RESET}"

# =============================================================================
# Check 1 — No phantom / stub directories in root
# =============================================================================
header "Check 1: No phantom/ghost directories or stubs in root"
run_check \
  "Phantom directories check (backend/, stubs/, core/ must not exist)" \
  bash -c 'for dir in backend stubs core; do
    if [ -d "$dir" ]; then
      echo "Found phantom directory: $dir"
    fi
  done'

# =============================================================================
# Check 2 — No foreign group publications in Tafkir modules
# =============================================================================
header "Check 2: No foreign group IDs declared in build files"
run_check \
  "Foreign group ID check (modules must not declare tech.kayys.alkhawarizm or tech.kayys.gollek)" \
  grep -rn --include="*.kts" \
    'group = "tech.kayys.alkhawarizm"\|group = "tech.kayys.gollek"' .

# =============================================================================
# Check 3 — No pseudo-project references to Alkhawarizm or backend
# =============================================================================
header "Check 3: Canonical coordinates used for Alkhawarizm dependencies"
run_check \
  "No project(\":core:alkhawarizm-*\") or project(\":backend:*\") references in build files" \
  grep -rn --include="*.kts" \
    'project(":core:alkhawarizm\|project(":backend:' \
    --exclude-dir="build" \
    --exclude="settings.gradle.kts" .

# =============================================================================
# Check 4 (WARN ONLY) — Check missing explicit group in published modules
# =============================================================================
header "Check 4 (warn only): All published modules declare group = \"tech.kayys.tafkir\""
run_warn_check \
  "Modules publishing Maven artifacts missing explicit group = \"tech.kayys.tafkir\"" \
  bash -c 'for kts in $(find . -maxdepth 4 -name "build.gradle.kts" | grep -v "/build/" | grep -v "/buildSrc/"); do
    if grep -q "maven-publish" "$kts" 2>/dev/null; then
      if ! grep -q "group = \"tech.kayys.tafkir\"" "$kts"; then
        echo "Missing explicit group: $kts"
      fi
    fi
  done'

# =============================================================================
# Summary
# =============================================================================
PASSED_CHECKS=$((TOTAL_CHECKS - FAILED_CHECKS))

echo
echo "${BOLD}=============================================================${RESET}"
if [[ ${FAILED_CHECKS} -eq 0 ]]; then
  echo "  ${GREEN}${BOLD}${PASSED_CHECKS}/${TOTAL_CHECKS} checks passed — all Tafkir boundaries clean.${RESET}"
  echo "${BOLD}=============================================================${RESET}"
  echo
  exit 0
else
  echo "  ${RED}${BOLD}${PASSED_CHECKS}/${TOTAL_CHECKS} checks passed — ${FAILED_CHECKS} boundary violation(s) found:${RESET}"
  for failure in "${FAILURES[@]}"; do
    echo "    ${RED}• ${failure}${RESET}"
  done
  echo "${BOLD}=============================================================${RESET}"
  echo
  exit 1
fi
