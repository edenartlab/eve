#!/usr/bin/env bash
# Deploy the generation-model-overhaul branch and push its tool definitions to
# the tools3 collection, then read them back to prove it took effect.
#
#   ./migration/deploy-generation-overhaul.sh STAGE     # do this first
#   ./migration/deploy-generation-overhaul.sh PROD      # after STAGE looks good
#
# IMPORTANT: tool definitions live in MongoDB (tools3), not in the yaml files.
# Editing api.yaml is inert until `eve tool update` runs. And the update must be
# run FROM THIS WORKTREE — running it from the main checkout would push that
# checkout's (stale) yamls instead. That is why this script pins sys.path.

set -euo pipefail

DB="${1:-STAGE}"
if [[ "$DB" != "STAGE" && "$DB" != "PROD" ]]; then
  echo "usage: $0 [STAGE|PROD]" >&2
  exit 1
fi

WORKTREE="/Users/gene/Dev/eden1/eve-overhaul"
PY="/Users/gene/Dev/eden1/eve/.venv/bin/python"
cd "$WORKTREE"

# Every tool whose api.yaml differs from main on this branch.
TOOLS=(
  create
  seedance2 seedance2_reference wan_27 vidu_reference gpt_image_2
  veo_31_lite flux_dev_fal kling_v3 kling_o3
  # retirements (active:false / visible:false must reach the DB too)
  flux_kontext kling_v25 kling kling_pro runway2 runway3 seedream3
  openai_image_generate openai_image_edit
)

echo "=============================================================="
echo " 1/3  Deploying code to $DB"
echo "=============================================================="
DB="$DB" modal deploy eve/api/api.py

echo
echo "=============================================================="
echo " 2/3  Pushing ${#TOOLS[@]} tool definitions to $DB tools3"
echo "=============================================================="
"$PY" -c "
import sys
sys.path.insert(0, '$WORKTREE')          # worktree yamls, not the main checkout
sys.argv = ['eve', 'tool', 'update', '--db', '$DB'] + '${TOOLS[*]}'.split()
from eve.cli import cli
cli()
"

echo
echo "=============================================================="
echo " 3/3  Reading back from $DB to verify"
echo "=============================================================="
"$PY" -c "
import sys
sys.path.insert(0, '$WORKTREE')
from eve import load_env
load_env('$DB')
from eve.tool import Tool

expect_active   = {'create','seedance2','seedance2_reference','wan_27',
                   'vidu_reference','gpt_image_2','veo_31_lite','kling_v3','kling_o3'}
expect_retired  = {'flux_kontext','kling_v25','kling','kling_pro','runway2',
                   'runway3','seedream3','openai_image_generate','openai_image_edit'}

fail = []
for key in sorted(expect_active | expect_retired):
    try:
        t = Tool.load(key)
    except Exception as e:
        fail.append(f'{key}: NOT IN DB ({e})'); continue
    active = getattr(t, 'active', None)
    if key in expect_active and not active:
        fail.append(f'{key}: expected active=True, got {active}')
    if key in expect_retired and active:
        fail.append(f'{key}: expected active=False (retired), got {active}')
    print(f'  {key:24s} active={active}  cost={getattr(t, \"cost_estimate\", None)}')

# spot-check the specific schema fixes actually landed in the DB
print()
checks = []
w = Tool.load('wan_27')
ar = (w.parameters or {}).get('aspect_ratio', {})
choices = ar.get('choices') or getattr(ar, 'choices', None)
checks.append(('wan_27 aspect_ratio has no \"auto\"', 'auto' not in (choices or [])))
checks.append(('wan_27 aspect_ratio keeps 4:3', '4:3' in (choices or [])))
s2 = Tool.load('seedance2')
checks.append(('seedance2 has no seed param', 'seed' not in (s2.parameters or {})))
v = Tool.load('vidu_reference')
checks.append(('vidu_reference present', v is not None))
for label, ok in checks:
    print(f'  [{\"OK\" if ok else \"FAIL\"}] {label}')
    if not ok:
        fail.append(label)

print()
if fail:
    print('VERIFICATION FAILED:'); [print('  -', f) for f in fail]; sys.exit(1)
print('All tools verified in $DB.')
"

echo
echo "Done. Next: smoke-test the new tools on STAGE, e.g."
echo "  eve tool test vidu_reference --db STAGE"
echo "  eve tool test wan_27 --db STAGE"
echo "(vidu_reference has never executed — run it at least once.)"
