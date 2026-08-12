#!/usr/bin/env bash
# check_rule_ancestry.sh -- re-runnable ancestry check for 02.7-SCREENING-RULE.md and its
# amendments.
#
# Plan 02.7-08 committed 02.7-SCREENING-RULE.md ALONE, before any synthetic benchmark
# cloud was scored (SC-1's structural gate). Plan 02.7-09 committed
# 02.7-SCREENING-RULE-AMENDMENT-01.md ALONE, amending the rule's D-12 local-dispersion
# gate, also before any benchmark cloud was scored. Plan 02.7-12 re-runs this exact check
# at phase close against every commit that scored a cloud, per the 02.6-07/02.6-15
# precedent (git merge-base --is-ancestor), extended here to cover both documents.
#
# Usage:
#   ./check_rule_ancestry.sh <RULE_SHA> [MEASUREMENT_SHA ...]
#   ./check_rule_ancestry.sh <RULE_SHA> --amendment <AMENDMENT_SHA> [MEASUREMENT_SHA ...]
#
# With no MEASUREMENT_SHA arguments, only asserts each document's own commit shape
# (touches exactly one file, no notebooks/ path) and that it is an ancestor of HEAD.
# With one or more MEASUREMENT_SHA arguments, additionally asserts each document's commit
# is an ancestor of each one -- i.e. it was committed before that measurement.
#
# The --amendment form additionally asserts AMENDMENT_SHA touches exactly
# 02.7-SCREENING-RULE-AMENDMENT-01.md, and that RULE_SHA is an ancestor of AMENDMENT_SHA
# (the amendment provably post-dates the document it amends) -- the same shape
# 02.5-PREREGISTRATION-AMENDMENT-01.md's own ancestry proof uses.
#
# Exits non-zero on the first failed assertion, printing which one failed.

set -u

RULE_SHA="${1:-}"
if [ -z "$RULE_SHA" ]; then
  echo "usage: $0 <RULE_SHA> [--amendment <AMENDMENT_SHA>] [MEASUREMENT_SHA ...]" >&2
  exit 2
fi
shift || true

AMENDMENT_SHA=""
if [ "${1:-}" = "--amendment" ]; then
  shift || true
  AMENDMENT_SHA="${1:-}"
  if [ -z "$AMENDMENT_SHA" ]; then
    echo "usage: $0 <RULE_SHA> --amendment <AMENDMENT_SHA> [MEASUREMENT_SHA ...]" >&2
    exit 2
  fi
  shift || true
fi

MEASUREMENT_SHAS=("$@")

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 2

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

check_document_shape() {
  # $1 = SHA, $2 = expected filename suffix (e.g. 02.7-SCREENING-RULE.md), $3 = label
  local sha="$1" expected_suffix="$2" label="$3"

  local file_count
  file_count="$(git show --name-only --format= "$sha" | grep -c .)"
  if [ "$file_count" != "1" ]; then
    fail "${label} commit ${sha} touches ${file_count} files, expected exactly 1"
  fi
  echo "PASS: ${label} commit touches exactly 1 file"

  local touched_file
  touched_file="$(git show --name-only --format= "$sha" | grep .)"
  case "$touched_file" in
    *"$expected_suffix") : ;;
    *) fail "${label} commit ${sha} touches ${touched_file}, expected ${expected_suffix}" ;;
  esac
  echo "PASS: the touched file is ${expected_suffix}"

  local notebook_count
  notebook_count="$(git show --name-only --format= "$sha" | grep -c "^notebooks/")"
  if [ "$notebook_count" != "0" ]; then
    fail "${label} commit ${sha} touches ${notebook_count} notebooks/ path(s), expected 0"
  fi
  echo "PASS: no notebooks/ path touched in the ${label} commit"

  if ! git merge-base --is-ancestor "$sha" HEAD; then
    fail "${label} commit ${sha} is not an ancestor of HEAD"
  fi
  echo "PASS: ${label} commit ${sha} is an ancestor of HEAD"
}

echo "== check_rule_ancestry.sh: RULE_SHA=${RULE_SHA} AMENDMENT_SHA=${AMENDMENT_SHA:-<none>} =="

# 1-4. The rule's own commit shape and ancestry.
check_document_shape "$RULE_SHA" "02.7-SCREENING-RULE.md" "rule"

# 5. The rule commit is an ancestor of every supplied measurement commit.
if [ "${#MEASUREMENT_SHAS[@]}" -gt 0 ]; then
  for sha in "${MEASUREMENT_SHAS[@]}"; do
    if ! git merge-base --is-ancestor "$RULE_SHA" "$sha"; then
      fail "rule commit ${RULE_SHA} is NOT an ancestor of measurement commit ${sha}"
    fi
    echo "PASS: rule commit ${RULE_SHA} is an ancestor of measurement commit ${sha}"
  done
fi

if [ -n "$AMENDMENT_SHA" ]; then
  # 6-9. The amendment's own commit shape and ancestry.
  check_document_shape "$AMENDMENT_SHA" "02.7-SCREENING-RULE-AMENDMENT-01.md" "amendment"

  # 10. The rule commit is an ancestor of the amendment commit -- the amendment provably
  # post-dates the document it amends (02.5-PREREGISTRATION-AMENDMENT-01.md's own shape).
  if ! git merge-base --is-ancestor "$RULE_SHA" "$AMENDMENT_SHA"; then
    fail "rule commit ${RULE_SHA} is NOT an ancestor of amendment commit ${AMENDMENT_SHA}"
  fi
  echo "PASS: rule commit ${RULE_SHA} is an ancestor of amendment commit ${AMENDMENT_SHA}"

  # 11. The amendment commit is an ancestor of every supplied measurement commit.
  if [ "${#MEASUREMENT_SHAS[@]}" -gt 0 ]; then
    for sha in "${MEASUREMENT_SHAS[@]}"; do
      if ! git merge-base --is-ancestor "$AMENDMENT_SHA" "$sha"; then
        fail "amendment commit ${AMENDMENT_SHA} is NOT an ancestor of measurement commit ${sha}"
      fi
      echo "PASS: amendment commit ${AMENDMENT_SHA} is an ancestor of measurement commit ${sha}"
    done
  fi
fi

echo "== all checks passed =="
exit 0
