#!/usr/bin/env bash
# check_rule_ancestry.sh -- re-runnable ancestry check for 02.7-SCREENING-RULE.md.
#
# Plan 02.7-08 committed 02.7-SCREENING-RULE.md ALONE, before any synthetic benchmark
# cloud was scored (SC-1's structural gate). Plan 02.7-12 re-runs this exact check at
# phase close against every commit that scored a cloud, per the 02.6-07/02.6-15
# precedent (git merge-base --is-ancestor).
#
# Usage:
#   ./check_rule_ancestry.sh <RULE_SHA> [MEASUREMENT_SHA ...]
#
# With no MEASUREMENT_SHA arguments, only asserts the rule's own commit shape
# (touches exactly one file, no notebooks/ path) and that it is an ancestor of HEAD.
# With one or more MEASUREMENT_SHA arguments, additionally asserts the rule commit is
# an ancestor of each one -- i.e. the rule was committed before that measurement.
#
# Exits non-zero on the first failed assertion, printing which one failed.

set -u

RULE_SHA="${1:-}"
if [ -z "$RULE_SHA" ]; then
  echo "usage: $0 <RULE_SHA> [MEASUREMENT_SHA ...]" >&2
  exit 2
fi
shift || true
MEASUREMENT_SHAS=("$@")

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 2

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

echo "== check_rule_ancestry.sh: RULE_SHA=${RULE_SHA} =="

# 1. The rule commit touches exactly one file.
FILE_COUNT="$(git show --name-only --format= "$RULE_SHA" | grep -c .)"
if [ "$FILE_COUNT" != "1" ]; then
  fail "rule commit ${RULE_SHA} touches ${FILE_COUNT} files, expected exactly 1"
fi
echo "PASS: rule commit touches exactly 1 file"

# 2. That one file is 02.7-SCREENING-RULE.md itself.
TOUCHED_FILE="$(git show --name-only --format= "$RULE_SHA" | grep .)"
case "$TOUCHED_FILE" in
  *02.7-SCREENING-RULE.md) : ;;
  *) fail "rule commit ${RULE_SHA} touches ${TOUCHED_FILE}, expected 02.7-SCREENING-RULE.md" ;;
esac
echo "PASS: the touched file is 02.7-SCREENING-RULE.md"

# 3. No notebooks/ path was touched in the rule commit.
NOTEBOOK_COUNT="$(git show --name-only --format= "$RULE_SHA" | grep -c "^notebooks/")"
if [ "$NOTEBOOK_COUNT" != "0" ]; then
  fail "rule commit ${RULE_SHA} touches ${NOTEBOOK_COUNT} notebooks/ path(s), expected 0"
fi
echo "PASS: no notebooks/ path touched in the rule commit"

# 4. The rule commit is an ancestor of HEAD.
if ! git merge-base --is-ancestor "$RULE_SHA" HEAD; then
  fail "rule commit ${RULE_SHA} is not an ancestor of HEAD"
fi
echo "PASS: rule commit ${RULE_SHA} is an ancestor of HEAD"

# 5. The rule commit is an ancestor of every supplied measurement commit.
if [ "${#MEASUREMENT_SHAS[@]}" -gt 0 ]; then
  for sha in "${MEASUREMENT_SHAS[@]}"; do
    if ! git merge-base --is-ancestor "$RULE_SHA" "$sha"; then
      fail "rule commit ${RULE_SHA} is NOT an ancestor of measurement commit ${sha}"
    fi
    echo "PASS: rule commit ${RULE_SHA} is an ancestor of measurement commit ${sha}"
  done
fi

echo "== all checks passed =="
exit 0
