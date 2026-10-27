#!/usr/bin/env bash
# Submit ONE CPU cache job, then 57 GPU jobs gated by afterok dependency.
set -euo pipefail
MODE=${1:-plan}
ROOT=$CANCER_DATA_ROOT
RUN="$ROOT/cancer_fullframe_cached_v1"
test -s "$RUN/jobs.tsv" && test -s "$RUN/pbs_prepare_cached.pbs" && test -s "$RUN/pbs_train_cached.pbs"
test "$(wc -l < "$RUN/jobs.tsv")" -eq 57
case "$MODE" in
 plan)
   echo 'NEW cache-backed 57-job campaign: 30 Core B0 + 12 Core A1 + 15 Angle.'
   echo 'CPU-only cache job on default queue (24h allocation), then GPU array.'
   echo 'IMPORTANT: cancel the OLD dependent GPU array 792737[] before submitting.'
   echo "Submit once: bash '$RUN/submit_cached_campaign.sh' all"
   ;;
 all)
   # Never race the already-submitted raw-TIFF GPU array.
   if qstat -u "$USER" | grep -Eq '792737(\[|\.)'; then
     echo 'STOP: old 792737 GPU array is still present. Cancel it first.' >&2
     exit 2
   fi
   if [[ -e "$RUN/cached_submissions.txt" ]]; then
     echo "Already submitted; see $RUN/cached_submissions.txt" >&2; exit 2
   fi
   # Single reservation lock against duplicate arrays from two shells.
   mkdir "$RUN/.submission_lock"
   CACHE_ID=$(qsub "$RUN/pbs_prepare_cached.pbs")
   echo "CACHE_JOB=$CACHE_ID" > "$RUN/cached_submissions.txt"
   ARRAY_ID=$(qsub -q gpus -W "depend=afterok:$CACHE_ID" -J "1-57" "$RUN/pbs_train_cached.pbs")
   echo "GPU_ARRAY=$ARRAY_ID" >> "$RUN/cached_submissions.txt"
   echo "Submitted CPU cache: $CACHE_ID"
   echo "Submitted 57 GPU jobs, held until cache validated: $ARRAY_ID"
   ;;
 *) echo 'Usage: submit_cached_campaign.sh plan|all' >&2; exit 2;;
esac
