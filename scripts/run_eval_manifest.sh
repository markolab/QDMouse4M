#!/usr/bin/env bash
set -euo pipefail

: "${QDMOUSE4M_RELEASE_ROOT:?Set QDMOUSE4M_RELEASE_ROOT.}"

MANIFEST="${1:-configs/eval_manifest.example.tsv}"

tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r model_kind model_name model_dir eval_root split out_dir; do
  [[ -z "${model_kind}" ]] && continue
  case "${eval_root}" in
    /*) eval_abs="${eval_root}" ;;
    *) eval_abs="${QDMOUSE4M_RELEASE_ROOT}/${eval_root}" ;;
  esac
  case "${model_kind}" in
    lp3d)
      bash scripts/run_lp3d_eval.sh "${model_dir}" "${eval_abs}" "${split}" "${out_dir}"
      ;;
    sleap)
      bash scripts/run_sleap_eval.sh "${model_dir}" "${eval_abs}" "${split}" "${out_dir}"
      ;;
    *)
      echo "Unknown model_kind in ${MANIFEST}: ${model_kind}" >&2
      exit 2
      ;;
  esac
done

