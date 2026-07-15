set -euo pipefail
export MPLBACKEND=Agg
unset DISPLAY

STAMP="$(date +%Y%m%d_%H%M%S)"
PACK="review_pack_${STAMP}"
mkdir -p "${PACK}"/{logs,main,external,posthoc,env}

# 0) 记录环境（非常有用：复现/审稿人也爱看）
{
  echo "=== date ==="; date
  echo "=== pwd ==="; pwd
  echo "=== python ==="; which python || true
  echo "=== python version ==="; python -V || true
  echo "=== pip freeze (top 200) ==="; pip freeze | head -n 200 || true
  echo "=== torch ==="; python - << 'PY' || true
import torch, sys
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY
} > "${PACK}/env/env.txt" 2>&1

# 1) 收集关键日志（看有没有 warning/隐藏失败）
for f in \
  1.sh.o 1.sh.e 2.sh.o 2.sh.e 3.sh.o 3.sh.e 4.sh.o 4.sh.e \
  5.sh.o 5.sh.e 6.sh.o 6.sh.e 7.sh.o 5_interp.e \
  8.sh.o 8_posthoc.o 8_posthoc.e \
  10_natmethods_analysis_upgrade.o 10_natmethods_analysis_upgrade.e \
  new-full.sh.o new-full.sh.e \
; do
  [ -f "$f" ] && cp -a "$f" "${PACK}/logs/"
done

# 2) 主数据（GSE83894 / out_plan8）关键 summary + ceiling + CI + 论文表
shopt -s nullglob
if [ -d out_plan8/summary_plan8 ]; then
  mkdir -p "${PACK}/main/summary_plan8"
  cp -a out_plan8/summary_plan8/*.{tsv,csv,json,md,png,pdf} "${PACK}/main/summary_plan8/" 2>/dev/null || true
fi
if [ -d out_plan8/summary_vnext ]; then
  mkdir -p "${PACK}/main/summary_vnext"
  cp -a out_plan8/summary_vnext/*.{tsv,csv,json,md,png,pdf} "${PACK}/main/summary_vnext/" 2>/dev/null || true
fi

# include main ceiling jsons for downstream unified ceiling benchmark
if [ -d out_plan8 ]; then
  mkdir -p "${PACK}/main/out_plan8"
  find out_plan8 -type f \(       -name "ceiling_formatA_gb_by_prefix.json" -o       -name "ceiling_formatA_gb.json" -o       -name "ceiling_formatA.json"     \) -print0 | while IFS= read -r -d '' f; do
      mkdir -p "${PACK}/main/$(dirname "$f")"
      cp -a "$f" "${PACK}/main/$f"
    done
fi

# learning curve & transfer（如果有）
for d in out_plan8/learning_curve out_plan8/transfer out_plan8/transfer_plan8 out_plan8/learning_curve_split0; do
  if [ -d "$d" ]; then
    mkdir -p "${PACK}/main/$(basename "$d")"
    find "$d" -maxdepth 2 -type f \( -name "*.tsv" -o -name "*.csv" -o -name "*.json" -o -name "*.md" -o -name "*.png" -o -name "*.pdf" \) \
      -exec cp -a {} "${PACK}/main/$(basename "$d")/" \; 2>/dev/null || true
  fi
done

# 3) 外部验证（GSE142696）最关键：按 trim 的 paper table + design_summary
if [ -d out_gse142696_plan8/design_summary ]; then
  mkdir -p "${PACK}/external/design_summary"
  cp -a out_gse142696_plan8/design_summary/*.{tsv,csv,json,md,png,pdf} "${PACK}/external/design_summary/" 2>/dev/null || true
fi

# 额外：每个 design/trim 的 design_gap_analysis.md & ceiling json（很关键，但体积不大）
if [ -d out_gse142696_plan8 ]; then
  mkdir -p "${PACK}/external/per_design_trim"
  find out_gse142696_plan8 -type f \( \
      -name "design_gap_analysis.md" -o \
      -name "ceiling_formatA_gb_by_prefix.json" -o \
      -name "bootstrap_ci_report.tsv" -o \
      -name "*paper_table*.tsv" \
    \) -print0 | while IFS= read -r -d '' f; do
      # 保留相对路径结构，便于我定位是哪一个 design/trim
      mkdir -p "${PACK}/external/per_design_trim/$(dirname "$f")"
      cp -a "$f" "${PACK}/external/per_design_trim/$f"
    done
fi

# 4) posthoc_natcomm_plus（你 8.sh 的核心产物：ceiling gap / binning / residual / negctrl / motif）
if [ -d posthoc_natcomm_plus ]; then
  mkdir -p "${PACK}/posthoc"
  find posthoc_natcomm_plus -type f \( -name "*.tsv" -o -name "*.csv" -o -name "*.json" -o -name "*.md" -o -name "*.png" -o -name "*.pdf" \) \
    -print0 | while IFS= read -r -d '' f; do
      mkdir -p "${PACK}/posthoc/$(dirname "$f")"
      cp -a "$f" "${PACK}/posthoc/$f"
    done
fi

# 4b) NatMethods Analysis 升级产物（10_natmethods_analysis_upgrade.sh）
if [ -d posthoc_natmethods_analysis ]; then
  mkdir -p "${PACK}/posthoc_natmethods"
  find posthoc_natmethods_analysis -type f \( -name "*.tsv" -o -name "*.csv" -o -name "*.json" -o -name "*.md" -o -name "*.png" -o -name "*.pdf" \) \
    -print0 | while IFS= read -r -d '' f; do
      mkdir -p "${PACK}/posthoc_natmethods/$(dirname "$f")"
      cp -a "$f" "${PACK}/posthoc_natmethods/$f"
    done
fi

# 4c) strong models 目录（可能会有 transformer / ms-resCNN 等的 summary，小而关键）
if [ -d out_plan8/strong_models ]; then
  mkdir -p "${PACK}/strong_models"
  find out_plan8/strong_models -type f \( -name "*.tsv" -o -name "*.csv" -o -name "*.json" -o -name "*.md" -o -name "*.png" -o -name "*.pdf" \) \
    -print0 | while IFS= read -r -d '' f; do
      mkdir -p "${PACK}/strong_models/$(dirname "$f")"
      cp -a "$f" "${PACK}/strong_models/$f"
    done
fi

if [ -d out_gse142696_plan8 ]; then
  find out_gse142696_plan8 -type d -name strong_models -print0 | while IFS= read -r -d '' d; do
    find "$d" -type f \( -name "*.tsv" -o -name "*.csv" -o -name "*.json" -o -name "*.md" -o -name "*.png" -o -name "*.pdf" \) \
      -print0 | while IFS= read -r -d '' f; do
        mkdir -p "${PACK}/strong_models/$(dirname "$f")"
        cp -a "$f" "${PACK}/strong_models/$f"
      done
  done
fi

# 4d) v4 patch outputs（11_run_nm_analysis_patch_v4.sh 的关键增强结果）
for d in nm_analysis_outputs_v4 nm_analysis_outputs_v4_final; do
  if [ -d "$d" ]; then
    mkdir -p "${PACK}/nm_patch/$(basename "$d")"
    find "$d" -type f \( -name "*.tsv" -o -name "*.csv" -o -name "*.json" -o -name "*.md" -o -name "*.png" -o -name "*.pdf" \) \
      -print0 | while IFS= read -r -d '' f; do
        mkdir -p "${PACK}/nm_patch/$(basename "$d")/$(dirname "$f")"
        cp -a "$f" "${PACK}/nm_patch/$(basename "$d")/$f"
      done
  fi
done

# 5) 写一个文件清单（我快速浏览用）
find "${PACK}" -type f | sort > "${PACK}/FILELIST.txt"

# 6) 打包
tar -czf "${PACK}.tar.gz" "${PACK}"
echo "DONE -> ${PACK}.tar.gz"
ls -lh "${PACK}.tar.gz"

