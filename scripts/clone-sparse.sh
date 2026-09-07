#!/bin/sh
# Blobless sparse clone of one ROCm monorepo subtree.
#
# rocm-systems and rocm-libraries are large monorepos, and each stage here needs
# two or three projects out of them. --filter=blob:none defers file content until
# checkout asks for it, so only the sparse set is ever fetched.
#
# usage: clone-sparse.sh <url> <dest> <commit-or-ref> <sparse-path>...
set -eu

url="$1"
dest="$2"
ref="$3"
shift 3

git clone --filter=blob:none --no-checkout "$url" "$dest"
git -C "$dest" sparse-checkout init --cone
git -C "$dest" sparse-checkout set "$@"
git -C "$dest" checkout "$ref"

echo "clone-sparse: $dest at $(git -C "$dest" rev-parse HEAD) ($ref)"
