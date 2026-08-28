#!/usr/bin/env bash
#
# The whole pipeline, end to end. Regenerates data/findings.json from
# data/raw/ on a clean checkout.
#
#   ./pipeline/run.sh
#
# Every step is resumable. A step whose output already exists is skipped, so an
# interrupted run picks up where it stopped rather than starting again.
#
# Four steps need a model. Each of those has two paths, in this order:
#
#   1. ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment. The step runs
#      itself and this script goes straight through.
#   2. Neither key. The step writes its prompt payloads, this script stops and
#      tells you which directory a Claude Code session has to fill, and you run
#      it again afterwards.
#
# Both paths write the same JSON to the same place and everything downstream
# validates it the same way. The committed results in this repository came from
# path 2.
#
# Fetching the discussion threads needs a GitHub token, because GitHub serves
# Discussions only through its GraphQL API and that API always requires one. Set
# GITHUB_TOKEN, or be logged in with the gh CLI. The threads are committed under
# data/raw/, so every step after the fetch runs with no token at all.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
note() { printf '   %s\n' "$1"; }

needs_model() {
  # $1 human name, $2 the module, $3 the run subcommand, $4 the directory a
  # person would have to fill, $5 the glob that says the step is finished.
  local name="$1" module="$2" sub="$3" dir="$4" glob="$5"
  if compgen -G "$dir/$glob" > /dev/null; then
    note "already answered: $(compgen -G "$dir/$glob" | wc -l | tr -d ' ') files in $dir"
    return 0
  fi
  if "$PY" "pipeline/$module" "$sub"; then
    return 0
  fi
  cat <<MSG

  STOP. $name needs a model and no API key is set.

  A Claude Code session should read each prompt payload and write the matching
  JSON answer file, then run this script again:

      in:  $(dirname "$dir")/in/
      out: $dir/

MSG
  exit 2
}

step "1. Fetch the corpus and the demand signal"
if [ -f data/raw/fetch_meta.json ]; then
  note "data/raw/ is already populated, skipping the fetch"
  note "delete data/raw/fetch_meta.json to refetch"
else
  "$PY" pipeline/fetch.py
fi

step "2. Split the pages into sections"
"$PY" pipeline/sections.py

step "3. Extract concepts and edges"
"$PY" pipeline/extract.py prep
needs_model "Concept extraction" extract.py run data/work/extract/out "batch-*.json"
"$PY" pipeline/extract.py assemble

step "4. Assemble the graph"
"$PY" pipeline/build_graph.py

step "5. Map the questions onto concepts"
"$PY" pipeline/extract.py prep-questions
needs_model "Question mapping" extract.py run-questions data/work/questions/out "qbatch-*.json"
"$PY" pipeline/extract.py assemble-questions
"$PY" pipeline/build_graph.py

step "6. Find F1, F2 and F3"
"$PY" pipeline/find.py

step "7. Name the missing fact on each near miss"
"$PY" pipeline/describe.py prep
needs_model "Gap descriptions" describe.py run data/work/describe/out "desc-*.json"
"$PY" pipeline/describe.py assemble

step "8. Retrieve both ways, generate, and score"
"$PY" pipeline/compare.py retrieve
needs_model "Vector path generation" compare.py run-vector data/work/compare/vector/out "gen-vector-*.json"
needs_model "Graph path generation" compare.py run-graph data/work/compare/graph/out "gen-graph-*.json"
"$PY" pipeline/compare.py prep-score
needs_model "Scoring" compare.py run-score data/work/compare/score/out "score-*.json"
"$PY" pipeline/compare.py assemble

step "9. Check a random sample of the findings"
"$PY" pipeline/validate.py prep
needs_model "Validation" validate.py run data/work/validate/out "val-*.json"
"$PY" pipeline/validate.py assemble

step "10. Write the manifest and the site payload"
"$PY" pipeline/manifest.py
"$PY" pipeline/site_data.py

step "Done"
note "data/findings.json, data/answers.json, data/validation.json, data/manifest.json"
note "open tool.html to browse the findings"
