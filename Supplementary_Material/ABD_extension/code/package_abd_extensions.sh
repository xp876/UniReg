#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$(pwd)}"
cd "$ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
PKG="abd_incremental_extension_${TS}"
mkdir -p "$PKG"
for rel in \
  out_gse142696_episomal_panel \
  out_variant_mpra_shared \
  out_variant_mpra_mech_validation \
  out_variant_mpra_transfer
 do
  if [[ -e "$rel" ]]; then
    cp -r "$rel" "$PKG/"
  fi
done

tar -czf "${PKG}.tar.gz" "$PKG"
rm -rf "$PKG"
echo "[done] wrote ${PKG}.tar.gz" >&2
