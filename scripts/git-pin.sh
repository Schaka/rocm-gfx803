#!/bin/sh
# Clone an upstream repository at an exactly-known commit.
#
# Every component here is pinned to a release *branch* (see README's
# "Component ref pinning"), and the Dockerfile is the source of truth for those
# branch names. But a branch cloned inside a RUN is invisible to the layer cache:
# the cache key is the command text, which does not change when AMD pushes a
# cherry-pick, so a build can silently reuse a layer from an older tip and nothing
# downstream knows. CI therefore resolves each branch to its commit once per run
# and passes it as <NAME>_SHA; this script performs the fetch so the stage sees
# one uniform interface, and keeps the branch-only form working for a manual
# build that supplies no SHA.
#
# The commit is fetched by name, because `--depth 1 --branch` cannot check out an
# arbitrary SHA and a full-history clone of pytorch to resolve one tip is not worth
# the fetch. So this does init, fetch, and checkout instead of git clone.
#
# usage: git-pin.sh <dest> <url> <ref> [sha]
set -eu

dest="$1"
url="$2"
ref="$3"
sha="${4:-}"

git init -q "$dest"
git -C "$dest" remote add origin "$url"

if [ -n "$sha" ]; then
    case "$sha" in
        *[!0-9a-fA-F]*) echo "git-pin: SHA argument is not hex: $sha" >&2; exit 1 ;;
    esac
    [ "${#sha}" -eq 40 ] || { echo "git-pin: expected a 40-char commit, got: $sha" >&2; exit 1; }
    git -C "$dest" fetch -q --depth 1 origin "$sha"
    git -C "$dest" checkout -q --detach FETCH_HEAD
else
    # No resolved commit, so follow the named ref itself. This is deliberately loud,
    # because a build that lands here compiles something nobody pinned by hand.
    echo "git-pin: WARNING: no *_SHA supplied for $ref; building whatever $url has for that ref right now." >&2
    # Most pins are release branches, but onnxruntime pins a release tag, and
    # `fetch refs/heads/v1.29.0` fails as a bare exit 128 that says nothing about the
    # ref type. Ask which one it is, then fetch that, so a plain local build does not
    # have to supply every *_SHA just to get past the clone.
    if git -C "$dest" ls-remote --exit-code --heads origin "refs/heads/$ref" >/dev/null 2>&1; then
        want="refs/heads/$ref"
    elif git -C "$dest" ls-remote --exit-code --tags origin "refs/tags/$ref" >/dev/null 2>&1; then
        want="refs/tags/$ref"
    else
        echo "git-pin: $url has neither a branch nor a tag named $ref" >&2
        exit 1
    fi
    # One fixed local name, because a ref like release/2.14 contains a slash and a
    # tag and a branch of the same name would otherwise land in different places.
    git -C "$dest" fetch -q --depth 1 origin "+$want:refs/remotes/origin/pinned"
    git -C "$dest" checkout -q --detach refs/remotes/origin/pinned
    echo "git-pin: $ref resolved to $want" >&2
fi

echo "git-pin: $dest at $(git -C "$dest" rev-parse HEAD) ($ref)"
