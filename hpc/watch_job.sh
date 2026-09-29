#!/bin/bash
# Poll a Slurm job from WSL until it reaches a terminal state, then dump its output.
# Written as a FILE rather than an inline ssh string: nested quoting through
# wsl -> bash -c -> ssh -> remote bash has corrupted three separate attempts tonight.
JOB="${1:?usage: watch_job.sh <jobid> <outfile> [errfile]}"
OUT="${2:?}"
ERR="${3:-}"

i=0
while [ "$i" -lt 80 ]; do
  state=$(timeout 40 ssh -o BatchMode=yes quartz "sacct -j $JOB --format=State -n 2>/dev/null | head -1" 2>/dev/null | tr -d ' ')
  case "$state" in
    COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*)
      echo "STATE=$state"
      break
      ;;
  esac
  i=$((i + 1))
  sleep 45
done

echo "=== OUT ==="
timeout 60 ssh -o BatchMode=yes quartz "cat $OUT 2>/dev/null"
if [ -n "$ERR" ]; then
  echo "=== ERR (tail) ==="
  timeout 60 ssh -o BatchMode=yes quartz "tail -12 $ERR 2>/dev/null"
fi
