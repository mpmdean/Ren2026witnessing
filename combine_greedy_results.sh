#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 NAME [OUTFILE]" >&2
  exit 2
fi

name=$1
outfile=${2:-${name}.txt}

# Escape regex metacharacters in $name so we match it literally
esc_name=$(printf '%s' "$name" | sed -e 's/[][(){}.^$*+?|\\]/\\&/g')

# Find candidate files (current dir; change -maxdepth or remove it if you want recursion),
# keep only exact basenames ^${name}[0-9]+_[0-9]+\.txt$, extract FIRST number, then sort & concat.
find . -maxdepth 1 -type f -name "${name}*.txt" -print0 \
| while IFS= read -r -d '' path; do
    base=${path##*/}
    if [[ $base =~ ^${esc_name}([0-9]+)_[0-9]+\.txt$ ]]; then
      num1=${BASH_REMATCH[1]}
      # Emit: NUMBER<tab>ABSOLUTE_PATH<nul>
      # (Use "$path" instead of readlink -f if you don’t care about absolute paths)
      printf '%s\t%s\0' "$num1" "$(readlink -f -- "$path")"
    fi
  done \
| sort -z -n -t $'\t' -k1,1 \
| awk -v RS='\0' -v ORS='\0' -F $'\t' '{print $2}' \
| xargs -0 -r cat -- > "$outfile"

#sed -i 's/[][]//g' "$outfile"
perl -0777 -i -ne 'while (/\[(.*?)\]/sg){ ($x=$1)=~s/\s+/ /g; print "$x\n"; }' "$outfile"

echo "Wrote: $outfile"
