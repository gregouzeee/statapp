#!/bin/bash
# Run all "fill the holes" scripts sequentially with separate log files.
# Usage:
#   bash run_all.sh
#
# Logs go to ./logs/ ; each script also resumes if the output JSONL
# already has rows, so re-running is safe (and free in API calls).
#
# Total expected runtime on Vertex AI Gemini 2.0 Flash:
#   - regenerate_triviaqa_num_topk.py : ~10 min  (≈3400 calls)
#   - regenerate_eli5_topk.py         : ~5 min   (≈100 calls)
#   - se_gsm8k.py                     : ~30-40 min (≈6600 calls)
#   - cross_table.py                  : ~1 min   (no API)
# So ~50 minutes end-to-end if run sequentially.

set -e
cd "$(dirname "$0")"
mkdir -p logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === Step 1/4: regenerate_triviaqa_num_topk.py ==="
python regenerate_triviaqa_num_topk.py --concurrency 8 2>&1 \
    | tee logs/regen_tqa_num.log

echo "[$(ts)] === Step 2/4: regenerate_eli5_topk.py ==="
python regenerate_eli5_topk.py --concurrency 4 2>&1 \
    | tee logs/regen_eli5.log

echo "[$(ts)] === Step 3/4: se_gsm8k.py ==="
python se_gsm8k.py --concurrency 4 2>&1 \
    | tee logs/se_gsm8k.log

echo "[$(ts)] === Step 4/4: cross_table.py ==="
python cross_table.py 2>&1 \
    | tee logs/cross_table.log

echo "[$(ts)] === ALL DONE ==="
echo "Outputs:"
ls -lh data/triviaqa_num_topk.jsonl data/eli5_topk.jsonl \
       data/se_gsm8k.jsonl data/cross_table.csv \
       data/cross_table_wide.csv data/cross_table.tex 2>/dev/null
