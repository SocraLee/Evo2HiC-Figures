# Shell counterpart of paths.py; the values come from paths.py itself (`python paths.py --sh`), so both read the
# same environment variables and defaults. Source it from a script in <section>/<item>/:
#   . "$(dirname "$0")/../../paths.sh"
PY=${PY:-python}            # interpreter of the Evo2HiC environment
eval "$("$PY" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.py" --sh)"
export PYTHONPATH=$PLOT_DIR:$REPO:${PYTHONPATH:-}   # evaluate/ (repo root) and the Evo2HiC modules
