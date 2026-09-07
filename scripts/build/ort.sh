#!/bin/sh
# Build the ONNX Runtime wheel with the MIGraphX execution provider.
set -eu

SRC=/onnxruntime
ARCH="${ROCM_ARCH:?ROCM_ARCH is required}"

# ORT's FetchContent for eigen pulls a gitlab archive tarball, which fails often
# enough to be worth doing as a shallow git fetch of the exact pinned commit
# instead.
eigen_commit="$(grep '^eigen;' "$SRC/cmake/deps.txt" | cut -d';' -f2 | grep -oP '(?<=archive/)[0-9a-f]{40}')"
mkdir -p /eigen-src
cd /eigen-src
git init -q
git remote add origin https://gitlab.com/libeigen/eigen.git
git fetch --depth 1 origin "$eigen_commit"
git checkout -q FETCH_HEAD

echo "ONNX Runtime build: arch $ARCH"
cd "$SRC"
python3 tools/ci_build/build.py \
    --config Release \
    --build_dir "$SRC/build" \
    --parallel \
    --build_wheel \
    --skip_tests \
    --allow_running_as_root \
    --compile_no_warning_as_error \
    --use_migraphx --migraphx_home /opt/rocm \
    --cmake_extra_defines "CMAKE_HIP_ARCHITECTURES=$ARCH" \
    --cmake_extra_defines "CMAKE_C_COMPILER_LAUNCHER=ccache" \
    --cmake_extra_defines "CMAKE_CXX_COMPILER_LAUNCHER=ccache" \
    --cmake_extra_defines "FETCHCONTENT_SOURCE_DIR_EIGEN=/eigen-src" \
    --cmake_extra_defines "CMAKE_POLICY_VERSION_MINIMUM=3.5" \
    --cmake_extra_defines "onnxruntime_USE_COMPOSABLE_KERNEL=OFF"

# Tag the wheel version after the build, not by editing VERSION_NUMBER first:
# onnxruntime_c_api.cc has a compile-time static_assert against a hardcoded
# literal that a PEP 440 local segment trips. The suffix matters because PyPI's
# own onnxruntime can report the exact version this build does, which would make
# the final image's exact-version constraint satisfiable from PyPI as well.
built_whl="$(ls "$SRC"/build/Release/dist/*.whl)"
unpack_dir="$SRC/build/Release/dist/unpacked"
python3 -m wheel unpack "$built_whl" -d "$unpack_dir"
old_dir="$(find "$unpack_dir" -mindepth 1 -maxdepth 1 -type d)"
old_name="$(basename "$old_dir")"
new_name="${old_name}+gfx803"
mv "$old_dir" "$unpack_dir/$new_name"
old_dist_info="$(find "$unpack_dir/$new_name" -maxdepth 1 -name '*.dist-info')"
new_dist_info="$unpack_dir/$new_name/${new_name}.dist-info"
mv "$old_dist_info" "$new_dist_info"
sed -i "s/^Version: .*/Version: ${old_name#*-}+gfx803/" "$new_dist_info/METADATA"
mkdir -p "$SRC/dist"
python3 -m wheel pack "$unpack_dir/$new_name" -d "$SRC/dist"
ls -l "$SRC/dist"
