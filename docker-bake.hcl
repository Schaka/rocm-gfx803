# Build graph for the gfx803 ROCm 10.0 pipeline. One target per component, one
# Dockerfile per target under docker/.
#
#   python-base ─ rocr-clr ─┬─ rocblas ─┬─ rocsolver ─┐
#                           │           ├─ sgemm-shim ┤
#                           │           └─────────────┼─ migraphx ─┬─ pytorch ─┬─ torchvision ─┐
#                           └─ miopen ────────────────┘            │           └─ torchaudio ──┤
#                                                                  ├─ ort ──────────────────────┤
#                                                                  └─ vllm ─────────────────────┴─ final
#                                                        triton ───────────────────────────────────┘
#
# This file is the single source of truth for image naming, tags, cache refs and
# every version knob. The Dockerfiles declare bare ARGs and get their values from
# here. The workflows set variables and name one target, and carry no build logic
# of their own.
#
# Local use:
#   docker buildx bake                        # build the whole graph
#   docker buildx bake rocblas                # one component and its dependencies
#   docker buildx bake --print final          # resolved graph, no build
#
# CI use: set the variables below as environment variables and name one target.

# ---------------------------------------------------------------------------
# Registry and naming
# ---------------------------------------------------------------------------

variable "REGISTRY" { default = "ghcr.io" }

# Lowercased repository owner. CI passes ${GITHUB_REPOSITORY_OWNER,,}, because
# ghcr package paths are case sensitive and always lower.
variable "OWNER" { default = "schaka" }

# These package names are shared with the mainline gfx900+ repo, where gfx803 is
# just another arch tag. Downstream consumers want final; the rest is build
# plumbing.
variable "PACKAGES" {
  default = {
    rocr-clr    = "rocm-rocr-clr-builder"
    rocblas     = "rocm-rocblas-builder"
    miopen      = "rocm-miopen-builder"
    rocsolver   = "rocm-rocsolver-builder"
    migraphx    = "rocm-migraphx-builder"
    pytorch     = "rocm-migraphx-torch-builder"
    torchvision = "rocm-torchvision-builder"
    torchaudio  = "rocm-torchaudio-builder"
    ort         = "rocm-migraphx-ort-builder"
    triton      = "rocm-triton-builder"
    vllm        = "rocm-vllm-builder"
    final       = "rocm-migraphx-ort-torch-builder"
  }
}

variable "ROCM_ARCH" { default = "gfx803" }

# Build line. Intermediate images are tagged :gfx803-rocm10 rather than :gfx803,
# because :gfx803 is what this repo published before the main line moved to 10.0.
# A component whose 10.0 job has not run since would otherwise be consumed as if
# it were a 10.0 build. The same value is stamped into every image at
# /opt/rocm/.gfx803-line and asserted by the stage that inherits it.
variable "LINE" { default = "rocm10" }

# The final image keeps the names people pull: latest-gfx803, rocm10.0-gfx803 and
# <date>-gfx803. The release already names the line, so it carries no LINE
# suffix.
variable "RELEASE_TAG" { default = "rocm10.0" }

# YYYYMMDD, for the dated tag. Empty publishes no dated tag rather than an
# invalid ":-gfx803".
variable "DATE" { default = "" }

# ---------------------------------------------------------------------------
# Build cache
# ---------------------------------------------------------------------------

# Exporting registry cache needs write access to the packages, which a local bake
# has no reason to have. CI sets this to true.
variable "REGISTRY_CACHE" { default = "false" }

# The component this run is actually building. Only that target exports cache. A
# dependency that bake happens to build as well must not write a cache tag under
# a package it publishes no image for.
variable "CACHE_TARGET" { default = "" }

# Force a fully fresh build. This drops cache-from as well as cache-to, so a bad
# layer already sitting under a cache ref cannot be reused. The run still
# repopulates the cache with its own result.
variable "NO_CACHE" { default = "false" }

# ---------------------------------------------------------------------------
# Component pins
#
# Branches, not commit SHAs, and no nightlies. Every upstream component is pinned
# to a named release branch, and CI resolves each branch to its commit once per
# run and passes it as <NAME>_SHA. A branch cloned inside a RUN is invisible to
# the layer cache, because the cache key is the command text, so without the
# resolved commit a build can silently reuse a layer from an older tip.
#
# Empty *_SHA means a manual build. scripts/git-pin.sh then follows the branch
# and says so on stderr.
# ---------------------------------------------------------------------------

variable "BASE_IMAGE" { default = "rocm/dev-ubuntu-26.04:10.0.0-full" }

# rocm-libraries and rocm-systems stopped cutting per-component rocm-rel-X.Y
# branches after 7.2. TheRock's release/therock-10.0 is the release line both
# repos track for 10.0.
variable "ROCM_SYSTEMS_REF"   { default = "release/therock-10.0" }
variable "ROCM_LIBRARIES_REF" { default = "release/therock-10.0" }
variable "ROCM_SYSTEMS_SHA"   { default = "" }
variable "ROCM_LIBRARIES_SHA" { default = "" }

# MIGraphX is still a standalone repo and still cuts its own
# release/rocm-rel-<major.minor> branch.
variable "MIGRAPHX_REF" { default = "release/rocm-rel-10.0" }
variable "MIGRAPHX_SHA" { default = "" }

# release/2.14 matches what the mainline repo pins for ROCm 10.0. The two
# companion refs come from that repo's own from-source fallback logic for torch
# 2.14, which is the path gfx803 always takes.
variable "PYTORCH_REF"     { default = "release/2.14" }
variable "TORCHVISION_REF" { default = "release/0.28" }
variable "TORCHAUDIO_REF"  { default = "release/2.11.0.2" }
variable "PYTORCH_SHA"     { default = "" }
variable "TORCHVISION_SHA" { default = "" }
variable "TORCHAUDIO_SHA"  { default = "" }

variable "ORT_VERSION" { default = "v1.29.0" }
variable "ORT_SHA"     { default = "" }

# Pinned to an exact commit, not a branch: triton has no release branch that
# tracks a given PyTorch version, and PyTorch itself pins triton to one exact
# commit per release for the same reason rocBLAS and MIOpen were pinned to a
# commit before the rocm-libraries monorepo restructure -- inductor's compiled
# extension API has to match exactly, not "close enough". This value is
# ROCm/pytorch release/2.14's own triton pin
# (.ci/docker/ci_commit_pins/triton.txt); bump it only together with
# PYTORCH_REF, to whatever that file names for the new ref.
variable "TRITON_REF" { default = "675c59878aa2280b31f722aaf42b825fcee21de8" }

# "auto" sizes the compile job count from MemAvailable. See
# scripts/lib/build-jobs.sh.
variable "BUILD_PARALLEL_LEVEL" { default = "auto" }

# TensorTopK.hip at -O3 has been measured taking 40GB of RSS and swap and several
# hours. CI passes -O1 for the pytorch job. Left at -O3 here because it stays a
# real performance tradeoff to opt into, not a fact about the file.
variable "TENSOR_TOPK_OPT_LEVEL" { default = "-O3" }

# Recorded inside every image at /opt/rocm/.gfx803-line and in its labels, so
# "which commits am I running?" is answerable from inside a container.
variable "GFX803_SOURCE_REV" { default = "unrecorded" }
variable "GFX803_PINS"       { default = "unrecorded" }

# ---------------------------------------------------------------------------
# Component wiring
#
# Each of these decides where a target gets a dependency from: the in-tree target,
# built in the same run, or the already published component image. CI sets the
# ones whose component ran as its own job, so nothing gets recompiled. A local
# build leaves them all false and builds the whole graph in one shot.
# ---------------------------------------------------------------------------

variable "WITH_ROCR_CLR_IMAGE"    { default = "false" }
variable "WITH_ROCBLAS_IMAGE"     { default = "false" }
variable "WITH_MIOPEN_IMAGE"      { default = "false" }
variable "WITH_ROCSOLVER_IMAGE"   { default = "false" }
variable "WITH_MIGRAPHX_IMAGE"    { default = "false" }
variable "WITH_PYTORCH_IMAGE"     { default = "false" }
variable "WITH_TORCHVISION_IMAGE" { default = "false" }
variable "WITH_TORCHAUDIO_IMAGE"  { default = "false" }
variable "WITH_ORT_IMAGE"         { default = "false" }
variable "WITH_TRITON_IMAGE"      { default = "false" }
variable "WITH_VLLM_IMAGE"        { default = "false" }

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------

function "pkg" {
  params = [component]
  result = "${REGISTRY}/${OWNER}/${PACKAGES[component]}"
}

function "image" {
  params = [component]
  result = "${pkg(component)}:${ROCM_ARCH}-${LINE}"
}

function "cache_ref" {
  params = [component]
  result = "${pkg(component)}:cache-${ROCM_ARCH}-${LINE}"
}

# pytorch, torchvision, torchaudio, ort, triton and vllm each carry a full build
# environment (a compiled ROCm SDK copy, a git checkout, a build tree, an LLVM
# build for triton) that final never reads -- it takes only the wheel(s) each one
# produces. Publishing that wheel alone under its own tag, from a `FROM scratch`
# stage named "wheels" in that component's own Dockerfile, is what final actually
# pulls; the plain image(component) tag above still publishes the full stage,
# because torchvision and torchaudio build against pytorch's full one. This is
# the fix for final running the runner disk out of space every time a new
# from-source component is wired into it: a COPY --from=<component> needs that
# component's whole published image on disk before it can copy one file out of
# it, so what final pulls has to be the wheel-only image.
function "wheels_image" {
  params = [component]
  result = "${pkg(component)}:${ROCM_ARCH}-${LINE}-wheels"
}

function "cache_from" {
  params = [component]
  result = REGISTRY_CACHE == "true" && NO_CACHE != "true" ? ["type=registry,ref=${cache_ref(component)}"] : []
}

# mode=max for the intermediates, because the next component's job imports them.
function "cache_to" {
  params = [component]
  result = REGISTRY_CACHE == "true" && CACHE_TARGET == component ? ["type=registry,ref=${cache_ref(component)},mode=max"] : []
}

# "target:<name>" builds the component here. "docker-image://<ref>" pulls the
# published one and drops that target out of the graph entirely. Buildx resolves
# the tag to its current digest on every build, so a republished component is
# picked up rather than served from an older cached layer.
function "ctx" {
  params = [component, use_image]
  result = use_image == "true" ? "docker-image://${image(component)}" : "target:${component}"
}

# Same as ctx(), but resolves to the trimmed "<component>-wheels" bake target /
# wheels_image() tag instead of the full one. Used only by final's own contexts
# for the six components that have a "wheels" stage.
function "wheels_ctx" {
  params = [component, use_image]
  result = use_image == "true" ? "docker-image://${wheels_image(component)}" : "target:${component}-wheels"
}

function "labels" {
  params = [component]
  result = {
    "io.rocm.gfx803.line"       = LINE
    "io.rocm.gfx803.stage"      = component
    "io.rocm.gfx803.source-rev" = GFX803_SOURCE_REV
    "io.rocm.gfx803.pins"       = GFX803_PINS
  }
}

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

group "default" {
  targets = ["final"]
}

# Never built. It exists so that `docker buildx bake --print pins` lists every
# version pin in one place, whatever components a given run happens to build.
# scripts/ci/resolve-pins.sh reads it to turn each branch into a commit.
target "pins" {
  dockerfile-inline = "FROM scratch"
  args = {
    BASE_IMAGE         = BASE_IMAGE
    ROCM_SYSTEMS_REF   = ROCM_SYSTEMS_REF
    ROCM_LIBRARIES_REF = ROCM_LIBRARIES_REF
    MIGRAPHX_REF       = MIGRAPHX_REF
    PYTORCH_REF        = PYTORCH_REF
    TORCHVISION_REF    = TORCHVISION_REF
    TORCHAUDIO_REF     = TORCHAUDIO_REF
    ORT_VERSION        = ORT_VERSION
    TRITON_REF         = TRITON_REF
  }
}

# Args are deliberately not set here. Buildx warns about a build-arg that no ARG
# in the Dockerfile consumes, so each target passes exactly what its own
# Dockerfile declares. That list doubles as a summary of what the stage depends
# on.
target "_common" {
  context  = "."
  attest   = ["type=provenance,disabled=true"]
  no-cache = NO_CACHE == "true"
}

# Every component stamps its line marker, so every component takes these three.
target "_stamped" {
  inherits = ["_common"]
  args = {
    GFX803_LINE       = LINE
    GFX803_SOURCE_REV = GFX803_SOURCE_REV
    GFX803_PINS       = GFX803_PINS
  }
}

# No registry cache: two cheap layers on top of BASE_IMAGE, and its own cache ref
# would mean a package tag whose contents depend on which BASE_IMAGE was passed.
target "python-base" {
  inherits   = ["_common"]
  dockerfile = "docker/python-base.Dockerfile"
  args = {
    BASE_IMAGE = BASE_IMAGE
  }
}

target "rocr-clr" {
  inherits   = ["_stamped"]
  dockerfile = "docker/rocr-clr.Dockerfile"
  contexts   = { python-base = "target:python-base" }
  args = {
    ROCM_SYSTEMS_REF     = ROCM_SYSTEMS_REF
    ROCM_SYSTEMS_SHA     = ROCM_SYSTEMS_SHA
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
  }
  labels     = labels("rocr-clr")
  tags       = [image("rocr-clr")]
  cache-from = cache_from("rocr-clr")
  cache-to   = cache_to("rocr-clr")
}

target "rocblas" {
  inherits   = ["_stamped"]
  dockerfile = "docker/rocblas.Dockerfile"
  contexts = {
    python-base = "target:python-base"
    rocr-clr    = ctx("rocr-clr", WITH_ROCR_CLR_IMAGE)
  }
  args = {
    ROCM_LIBRARIES_REF   = ROCM_LIBRARIES_REF
    ROCM_LIBRARIES_SHA   = ROCM_LIBRARIES_SHA
    ROCM_ARCH            = ROCM_ARCH
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
  }
  labels     = labels("rocblas")
  tags       = [image("rocblas")]
  cache-from = cache_from("rocblas")
  cache-to   = cache_to("rocblas")
}

target "miopen" {
  inherits   = ["_stamped"]
  dockerfile = "docker/miopen.Dockerfile"
  contexts = {
    python-base = "target:python-base"
    rocr-clr    = ctx("rocr-clr", WITH_ROCR_CLR_IMAGE)
  }
  args = {
    ROCM_LIBRARIES_REF   = ROCM_LIBRARIES_REF
    ROCM_LIBRARIES_SHA   = ROCM_LIBRARIES_SHA
    ROCM_ARCH            = ROCM_ARCH
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
  }
  labels     = labels("miopen")
  tags       = [image("miopen")]
  cache-from = cache_from("miopen")
  cache-to   = cache_to("miopen")
}

target "rocsolver" {
  inherits   = ["_stamped"]
  dockerfile = "docker/rocsolver.Dockerfile"
  contexts = {
    python-base = "target:python-base"
    rocblas     = ctx("rocblas", WITH_ROCBLAS_IMAGE)
  }
  args = {
    ROCM_LIBRARIES_REF   = ROCM_LIBRARIES_REF
    ROCM_LIBRARIES_SHA   = ROCM_LIBRARIES_SHA
    ROCM_ARCH            = ROCM_ARCH
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
  }
  labels     = labels("rocsolver")
  tags       = [image("rocsolver")]
  cache-from = cache_from("rocsolver")
  cache-to   = cache_to("rocsolver")
}

# Never published and never cached: one hipcc call that final always runs against
# the tree it was built from.
target "sgemm-shim" {
  inherits   = ["_common"]
  dockerfile = "docker/sgemm-shim.Dockerfile"
  contexts   = { rocblas = ctx("rocblas", WITH_ROCBLAS_IMAGE) }
  args = {
    ROCM_ARCH = ROCM_ARCH
  }
}

target "migraphx" {
  inherits   = ["_stamped"]
  dockerfile = "docker/migraphx.Dockerfile"
  contexts = {
    python-base = "target:python-base"
    rocblas     = ctx("rocblas", WITH_ROCBLAS_IMAGE)
    miopen      = ctx("miopen", WITH_MIOPEN_IMAGE)
    rocsolver   = ctx("rocsolver", WITH_ROCSOLVER_IMAGE)
  }
  args = {
    ROCM_ARCH            = ROCM_ARCH
    MIGRAPHX_REF         = MIGRAPHX_REF
    MIGRAPHX_SHA         = MIGRAPHX_SHA
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
  }
  labels     = labels("migraphx")
  tags       = [image("migraphx")]
  cache-from = cache_from("migraphx")
  cache-to   = cache_to("migraphx")
}

target "pytorch" {
  inherits   = ["_stamped"]
  dockerfile = "docker/pytorch.Dockerfile"
  target     = "builder"
  contexts = {
    python-base = "target:python-base"
    migraphx    = ctx("migraphx", WITH_MIGRAPHX_IMAGE)
  }
  args = {
    ROCM_ARCH              = ROCM_ARCH
    PYTORCH_REF            = PYTORCH_REF
    PYTORCH_SHA            = PYTORCH_SHA
    BUILD_PARALLEL_LEVEL   = BUILD_PARALLEL_LEVEL
    TENSOR_TOPK_OPT_LEVEL  = TENSOR_TOPK_OPT_LEVEL
  }
  labels     = labels("pytorch")
  tags       = [image("pytorch")]
  cache-from = cache_from("pytorch")
  cache-to   = cache_to("pytorch")
}

# The trimmed wheel-only companion. See the wheels_image() comment above.
target "pytorch-wheels" {
  inherits   = ["pytorch"]
  target     = "wheels"
  tags       = [wheels_image("pytorch")]
  cache-from = []
  cache-to   = []
}

target "torchvision" {
  inherits   = ["_stamped"]
  dockerfile = "docker/torchvision.Dockerfile"
  target     = "builder"
  contexts   = { pytorch = ctx("pytorch", WITH_PYTORCH_IMAGE) }
  args = {
    ROCM_ARCH       = ROCM_ARCH
    TORCHVISION_REF = TORCHVISION_REF
    TORCHVISION_SHA = TORCHVISION_SHA
  }
  labels     = labels("torchvision")
  tags       = [image("torchvision")]
  cache-from = cache_from("torchvision")
  cache-to   = cache_to("torchvision")
}

# The trimmed wheel-only companion. See the wheels_image() comment above.
target "torchvision-wheels" {
  inherits   = ["torchvision"]
  target     = "wheels"
  tags       = [wheels_image("torchvision")]
  cache-from = []
  cache-to   = []
}

target "torchaudio" {
  inherits   = ["_stamped"]
  dockerfile = "docker/torchaudio.Dockerfile"
  target     = "builder"
  contexts   = { pytorch = ctx("pytorch", WITH_PYTORCH_IMAGE) }
  args = {
    ROCM_ARCH      = ROCM_ARCH
    TORCHAUDIO_REF = TORCHAUDIO_REF
    TORCHAUDIO_SHA = TORCHAUDIO_SHA
  }
  labels     = labels("torchaudio")
  tags       = [image("torchaudio")]
  cache-from = cache_from("torchaudio")
  cache-to   = cache_to("torchaudio")
}

# The trimmed wheel-only companion. See the wheels_image() comment above.
target "torchaudio-wheels" {
  inherits   = ["torchaudio"]
  target     = "wheels"
  tags       = [wheels_image("torchaudio")]
  cache-from = []
  cache-to   = []
}

target "ort" {
  inherits   = ["_stamped"]
  dockerfile = "docker/ort.Dockerfile"
  target     = "builder"
  contexts = {
    python-base = "target:python-base"
    migraphx    = ctx("migraphx", WITH_MIGRAPHX_IMAGE)
  }
  args = {
    ROCM_ARCH   = ROCM_ARCH
    ORT_VERSION = ORT_VERSION
    ORT_SHA     = ORT_SHA
  }
  labels     = labels("ort")
  tags       = [image("ort")]
  cache-from = cache_from("ort")
  cache-to   = cache_to("ort")
}

# The trimmed wheel-only companion. See the wheels_image() comment above.
target "ort-wheels" {
  inherits   = ["ort"]
  target     = "wheels"
  tags       = [wheels_image("ort")]
  cache-from = []
  cache-to   = []
}

# Independent of every other target: it links no ROCm library at build time, so
# it takes only python-base, not rocblas/miopen/migraphx. The wheel it produces
# resolves libamdhip64 and libhsa-runtime64 by dlopen at runtime, against
# whatever the final image provides.
#
# Inherits _common, not _stamped: it carries no /opt/rocm tree forward (see
# docker/triton.Dockerfile), so it declares none of _stamped's three ARGs, and
# passing them anyway would be exactly the unconsumed-build-arg noise the
# _common comment above warns about. labels() still applies directly, since
# image labels are a bake-level annotation, not a Dockerfile ARG.
target "triton" {
  inherits   = ["_common"]
  dockerfile = "docker/triton.Dockerfile"
  target     = "builder"
  contexts   = { python-base = "target:python-base" }
  args = {
    TRITON_REF           = TRITON_REF
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
  }
  labels     = labels("triton")
  tags       = [image("triton")]
  cache-from = cache_from("triton")
  cache-to   = cache_to("triton")
}

# The trimmed wheel-only companion. See the wheels_image() comment above.
target "triton-wheels" {
  inherits   = ["triton"]
  target     = "wheels"
  tags       = [wheels_image("triton")]
  cache-from = []
  cache-to   = []
}

# The gfx803 vLLM fork lives in-tree at vllm/ (this repo tracks the tree
# directly, with no submodule and no separate history), so this target takes no
# *_REF pin and clones nothing:
# the build context already is the pinned source.
#
# Independent of the rocBLAS/MIOpen/rocSOLVER/MIGraphX chain, the same way
# triton is: vllm/CMakeLists.txt links only libamdhip64 at build time (see the
# comment on that COPY in docker/vllm.Dockerfile), so this target needs
# python-base's own stock ROCm for hipcc and the HIP headers, not migraphx's
# patched tree. It does need pytorch's wheel, because vllm/setup.py imports
# torch and compiles its extension against it.
target "vllm" {
  inherits   = ["_common"]
  dockerfile = "docker/vllm.Dockerfile"
  target     = "builder"
  contexts = {
    python-base = "target:python-base"
    pytorch     = wheels_ctx("pytorch", WITH_PYTORCH_IMAGE)
  }
  args = {
    ROCM_ARCH            = ROCM_ARCH
    BUILD_PARALLEL_LEVEL = BUILD_PARALLEL_LEVEL
  }
  labels     = labels("vllm")
  tags       = [image("vllm")]
  cache-from = cache_from("vllm")
  cache-to   = cache_to("vllm")
}

# The trimmed wheel-only companion. See the wheels_image() comment above.
target "vllm-wheels" {
  inherits   = ["vllm"]
  target     = "wheels"
  tags       = [wheels_image("vllm")]
  cache-from = []
  cache-to   = []
}

# No cache in either direction. Nothing builds from the final image, so its cache
# has no consumer, and every expensive component arrives prebuilt. Exporting
# anyway costs a second compressed copy of every layer, written to the same
# nearly full runner disk this job has run out of before.
target "final" {
  inherits   = ["_stamped"]
  dockerfile = "docker/final.Dockerfile"
  contexts = {
    python-base = "target:python-base"
    rocr-clr    = ctx("rocr-clr", WITH_ROCR_CLR_IMAGE)
    miopen      = ctx("miopen", WITH_MIOPEN_IMAGE)
    rocsolver   = ctx("rocsolver", WITH_ROCSOLVER_IMAGE)
    sgemm-shim  = "target:sgemm-shim"
    migraphx    = ctx("migraphx", WITH_MIGRAPHX_IMAGE)
    # These six take the trimmed "-wheels" companion, not the full build image:
    # final reads only /wheels (/onnxruntime/dist for ort) from each of them, and
    # the full images carry a compiled ROCm SDK copy, a git checkout, a build
    # tree, or (triton) a from-source LLVM/MLIR build that final has no use for.
    # See the wheels_image() comment in this file's derived-values section.
    pytorch     = wheels_ctx("pytorch", WITH_PYTORCH_IMAGE)
    torchvision = wheels_ctx("torchvision", WITH_TORCHVISION_IMAGE)
    torchaudio  = wheels_ctx("torchaudio", WITH_TORCHAUDIO_IMAGE)
    ort         = wheels_ctx("ort", WITH_ORT_IMAGE)
    triton      = wheels_ctx("triton", WITH_TRITON_IMAGE)
    vllm        = wheels_ctx("vllm", WITH_VLLM_IMAGE)
  }
  labels = labels("final")
  tags = concat(
    [
      "${pkg("final")}:latest-${ROCM_ARCH}",
      "${pkg("final")}:${RELEASE_TAG}-${ROCM_ARCH}",
    ],
    DATE != "" ? ["${pkg("final")}:${DATE}-${ROCM_ARCH}"] : [],
  )
}
