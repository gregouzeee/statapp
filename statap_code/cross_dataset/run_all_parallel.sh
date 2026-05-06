#!/bin/bash
# Run the 3 generation scripts IN PARALLEL, then build the cross table.
# Usage:
#   bash run_all_parallel.sh
#
# All three generation jobs are independent; they share the Vertex AI
# rate limit only. Adjust per-script --concurrency below if you hit
# 429s in the logs (lower numbers = friendlier to the quota).
#
# Total runtime: ~30-40 min (bound by se_gsm8k.py).

set -e
cd "$(dirname "$0")"
mkdir -p logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === Launching 3 generation jobs in parallel ==="
python regenerate_triviaqa_num_topk.py --concurrency 6 \
    > logs/regen_tqa_num.log 2>&1 &
PID1=$!
echo "  [pid $PID1] regenerate_triviaqa_num_topk.py"

python regenerate_eli5_topk.py --concurrency 3 \
    > logs/regen_eli5.log 2>&1 &
PID2=$!
echo "  [pid $PID2] regenerate_eli5_topk.py"

python se_gsm8k.py --num 300 --concurrency 3 \
    > logs/se_gsm8k.log 2>&1 &
PID3=$!
echo "  [pid $PID3] se_gsm8k.py (n=300)"

echo "[$(ts)] Waiting for the 3 jobs to finish... "
echo "        Tail their logs to follow progress, e.g.:"
echo "          tail -f logs/regen_tqa_num.log"
echo "          tail -f logs/regen_eli5.log"
echo "          tail -f logs/se_gsm8k.log"

wait $PID1 && echo "[$(ts)] regen_tqa_num done"   || echo "[$(ts)] regen_tqa_num FAILED"
wait $PID2 && echo "[$(ts)] regen_eli5 done"      || echo "[$(ts)] regen_eli5 FAILED"
wait $PID3 && echo "[$(ts)] se_gsm8k done"        || echo "[$(ts)] se_gsm8k FAILED"

echo "[$(ts)] === Building cross table ==="
python cross_table.py 2>&1 | tee logs/cross_table.log

echo "[$(ts)] === ALL DONE ==="
ls -lh data/triviaqa_num_topk.jsonl data/eli5_topk.jsonl \
       data/se_gsm8k.jsonl data/cross_table.csv \
       data/cross_table_wide.csv data/cross_table.tex 2>/dev/null
