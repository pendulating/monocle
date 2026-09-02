#!/usr/bin/env bash
#
# setup_env.sh - build a sudo-free local toolchain for the trashbot annotator.
#
# Stages:
#   1. ffmpeg    - use the zoo tree when usable, else vendor FFmpeg n7.1.
#   2. opencv    - build OpenCV 4.10.0 with FFmpeg-backed videoio.
#   3. videopipe - clone VideoPipe, apply patches, build libvideo_pipe.
#   4. verify    - check artifacts, library resolution, LLM symbols, decode.
#
# Usage: bash setup_env.sh [stage ...]   # default: all stages in order
#
# The script needs no sudo and no conda. Each stage skips when its output
# exists, so reruns are cheap. VideoPipe patches live as static files in
# third_party/patches/. The script applies them after each fresh clone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD="$SCRIPT_DIR/third_party"
SRC_DIR="$THIRD/src"
PATCH_DIR="$THIRD/patches"
LOG_DIR="$THIRD/logs"
VERSIONS_ENV="$THIRD/versions.env"

ZOO_FFMPEG="${TBA_ZOO_FFMPEG:-/share/pierson/matt/zoo/ffmpeg-libs/n7.1}"
FFMPEG_VER="7.1"
OPENCV_VER="4.10.0"
VP_REPO="https://github.com/sherlockchou86/VideoPipe.git"
VP_DIR="$THIRD/VideoPipe"
VP_BUILD_DIR="$THIRD/videopipe/build"
OCV_SRC_DIR="$SRC_DIR/opencv-$OPENCV_VER"
OCV_BUILD_DIR="$THIRD/opencv/build"
OCV_INSTALL_DIR="$THIRD/opencv/install"
FFMPEG_INSTALL_DIR="$THIRD/ffmpeg/install"
FFMPEG_BUILD_DIR="$THIRD/ffmpeg/build"
# Static patch files. setup_env.sh applies them in this order after a clone.
PATCH_FILES=(
    vp-gstreamer-optional.patch
    vp-gstreamer-fallback.patch
    vp-des-flush.patch
)
VP_GST="OFF"     # final VP_WITH_GSTREAMER value
SAMPLES="OFF"    # final VP_BUILD_SAMPLES value
DECODE_TEST_VIDEO="${TBA_DECODE_TEST_VIDEO:-/share/ju/robot_norms/data/Bu_TrashbotWoz/2022-09-28-astor-place/onboard_camera/recycling.mp4}"
JOBS="$(nproc)"

mkdir -p "$SRC_DIR" "$PATCH_DIR" "$LOG_DIR"

log()  { printf '[setup_env] %s\n' "$*"; }
warn() { printf '[setup_env] WARN: %s\n' "$*" >&2; }
die()  { printf '[setup_env] ERROR: %s\n' "$*" >&2; exit 1; }

# ver_ge A B: return 0 when version string A >= version string B.
ver_ge() {
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

require_tools() {
    local t
    for t in curl tar unzip git cmake pkg-config make awk g++ nm; do
        command -v "$t" >/dev/null 2>&1 || die "missing tool: $t"
    done
}

# run_logged LOGFILE CMD...: run CMD and send output to LOGFILE.
# On failure print the log tail and return the exit code.
run_logged() {
    local lf="$1"; shift
    local rc=0
    set +e
    "$@" >"$lf" 2>&1
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        warn "command failed with rc=$rc: $*"
        warn "log file: $lf"
        tail -n 40 "$lf" >&2 || true
    fi
    return "$rc"
}

# ---------------------------------------------------------------------------
# Stage 1: ffmpeg
# ---------------------------------------------------------------------------
stage_ffmpeg() {
    log "STAGE 1/4 ffmpeg"
    if [ -f "$ZOO_FFMPEG/lib/pkgconfig/libavcodec.pc" ] \
        && PKG_CONFIG_PATH="$ZOO_FFMPEG/lib/pkgconfig" pkg-config --exists libavcodec; then
        FFMPEG_ROOT="$ZOO_FFMPEG"
        FFMPEG_SOURCE="zoo"
        FFMPEG_PC_DIR="$FFMPEG_ROOT/lib/pkgconfig"
        FFMPEG_LIB_VERSION="$(PKG_CONFIG_PATH="$FFMPEG_PC_DIR" pkg-config --modversion libavcodec)"
        log "  use zoo tree at $FFMPEG_ROOT (libavcodec $FFMPEG_LIB_VERSION)"
        return 0
    fi

    warn "zoo tree not usable at $ZOO_FFMPEG; vendor FFmpeg $FFMPEG_VER"
    FFMPEG_ROOT="$FFMPEG_INSTALL_DIR"
    FFMPEG_SOURCE="vendored"
    FFMPEG_PC_DIR="$FFMPEG_ROOT/lib/pkgconfig"

    if [ -f "$FFMPEG_PC_DIR/libavcodec.pc" ]; then
        FFMPEG_LIB_VERSION="$(PKG_CONFIG_PATH="$FFMPEG_PC_DIR" pkg-config --modversion libavcodec)"
        log "  skip build; install exists at $FFMPEG_ROOT"
        return 0
    fi

    local tarball="$SRC_DIR/ffmpeg-$FFMPEG_VER.tar.xz"
    local srcdir="$SRC_DIR/ffmpeg-$FFMPEG_VER"
    if [ ! -d "$srcdir" ]; then
        log "  download https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VER.tar.xz"
        curl -fL --retry 3 -o "$tarball" "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VER.tar.xz"
        tar -xJf "$tarball" -C "$SRC_DIR"
    fi
    mkdir -p "$FFMPEG_BUILD_DIR"
    local conf_extra=()
    command -v nasm >/dev/null 2>&1 || conf_extra+=(--disable-x86asm)
    # Minimal build: shared libs only, no docs, no programs. Protocols stay on.
    # TLS stays off because we decode local files only.
    log "  configure minimal shared build"
    run_logged "$LOG_DIR/ffmpeg-configure.log" \
        "$srcdir/configure" \
        --prefix="$FFMPEG_INSTALL_DIR" \
        --enable-shared --disable-static \
        --disable-doc --disable-programs \
        "${conf_extra[@]}"
    run_logged "$LOG_DIR/ffmpeg-make.log" make -C "$FFMPEG_BUILD_DIR" -j"$JOBS"
    run_logged "$LOG_DIR/ffmpeg-install.log" make -C "$FFMPEG_BUILD_DIR" install
    FFMPEG_LIB_VERSION="$(PKG_CONFIG_PATH="$FFMPEG_PC_DIR" pkg-config --modversion libavcodec)"
    log "  installed FFmpeg at $FFMPEG_ROOT (libavcodec $FFMPEG_LIB_VERSION)"
}

# ---------------------------------------------------------------------------
# Stage 2: opencv
# ---------------------------------------------------------------------------
stage_opencv() {
    log "STAGE 2/4 opencv"
    # Point pkg-config at the chosen FFmpeg so WITH_FFMPEG finds libav*.
    export PKG_CONFIG_PATH="$FFMPEG_PC_DIR${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

    if [ -f "$OCV_INSTALL_DIR/lib/pkgconfig/opencv4.pc" ]; then
        local have
        have="$(PKG_CONFIG_PATH="$OCV_INSTALL_DIR/lib/pkgconfig" pkg-config --modversion opencv4)"
        if ver_ge "$have" "4.6"; then
            log "  skip build; opencv4 $have at $OCV_INSTALL_DIR"
            return 0
        fi
        warn "found opencv4 $have (< 4.6); rebuild"
    fi

    local zipball="$SRC_DIR/opencv-$OPENCV_VER.zip"
    if [ ! -d "$OCV_SRC_DIR" ]; then
        log "  download OpenCV $OPENCV_VER from GitHub releases"
        curl -fL --retry 3 -o "$zipball" \
            "https://github.com/opencv/opencv/archive/refs/tags/$OPENCV_VER.zip"
        unzip -q -o "$zipball" -d "$SRC_DIR"
    fi

    # VideoPipe needs the freetype module. It lives in opencv_contrib.
    local contrib_dir="$SRC_DIR/opencv_contrib-$OPENCV_VER"
    if [ ! -d "$contrib_dir" ]; then
        log "  download opencv_contrib $OPENCV_VER from GitHub releases"
        curl -fL --retry 3 -o "$SRC_DIR/opencv_contrib-$OPENCV_VER.zip" \
            "https://github.com/opencv/opencv_contrib/archive/refs/tags/$OPENCV_VER.zip"
        unzip -q -o "$SRC_DIR/opencv_contrib-$OPENCV_VER.zip" -d "$SRC_DIR"
    fi

    mkdir -p "$OCV_BUILD_DIR"
    log "  configure OpenCV (limited BUILD_LIST, FFmpeg through pkg-config)"
    # VideoPipe sources include opencv2/dnn.hpp and opencv2/freetype.hpp with
    # no guards. Build dnn from the main tree and freetype from contrib.
    # FreeType and HarfBuzz come from the host.
    run_logged "$LOG_DIR/opencv-cmake.log" cmake -S "$OCV_SRC_DIR" -B "$OCV_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$OCV_INSTALL_DIR" \
        -DCMAKE_PREFIX_PATH="$FFMPEG_ROOT" \
        -DOPENCV_EXTRA_MODULES_PATH="$SRC_DIR/opencv_contrib-$OPENCV_VER/modules" \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,video,dnn,freetype \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GTK=OFF \
        -DWITH_QT=OFF \
        -DWITH_IPP=OFF \
        -DWITH_OPENCL=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DOPENCV_GENERATE_PKGCONFIG=ON
    grep -Eq "FFMPEG:[[:space:]]+YES" "$LOG_DIR/opencv-cmake.log" \
        || die "OpenCV did not detect FFmpeg; see $LOG_DIR/opencv-cmake.log"
    log "  build with $JOBS jobs (this takes a while)"
    run_logged "$LOG_DIR/opencv-make.log" cmake --build "$OCV_BUILD_DIR" -j"$JOBS"
    run_logged "$LOG_DIR/opencv-install.log" cmake --install "$OCV_BUILD_DIR"

    local ver
    ver="$(PKG_CONFIG_PATH="$OCV_INSTALL_DIR/lib/pkgconfig" pkg-config --modversion opencv4)"
    ver_ge "$ver" "4.6" || die "opencv4 $ver < 4.6 after install"
    log "  installed opencv4 $ver at $OCV_INSTALL_DIR"
}

# ---------------------------------------------------------------------------
# Stage 3: videopipe
# ---------------------------------------------------------------------------
# patch_markers NAME: print "file:needle" entries. All needles must appear in
# the named files when the patch is already applied.
patch_markers() {
    case "$1" in
        vp-gstreamer-optional.patch)
            # Root CMake: option, sample switch, gst define, source exclusion.
            printf '%s\n' \
                "CMakeLists.txt:option(VP_WITH_GSTREAMER" \
                "CMakeLists.txt:option(VP_BUILD_SAMPLES" \
                "CMakeLists.txt:add_definitions(-DVP_HAS_GSTREAMER)" \
                "CMakeLists.txt:REMOVE_ITEM" \
                "CMakeLists.txt:if(VP_BUILD_SAMPLES)" \
                "samples/CMakeLists.txt:if(VP_WITH_GSTREAMER)"
            ;;
        vp-gstreamer-fallback.patch)
            # Node sources: CAP_ANY and fourcc fallbacks behind the macro.
            printf '%s\n' \
                "nodes/vp_file_src_node.cpp:VP_HAS_GSTREAMER" \
                "nodes/vp_file_des_node.cpp:VP_HAS_GSTREAMER"
            ;;
        vp-des-flush.patch)
            # Sink: release the writer on SPEAK so the mp4 finalizes.
            printf '%s\n' \
                "nodes/vp_file_des_node.cpp:vp-des-flush patch"
            ;;
        *) return 1 ;;
    esac
}

# patch_applied NAME: return 0 when every marker of the patch is present.
patch_applied() {
    local entry file needle
    # Markers hold no spaces, so word split on newlines is safe here.
    for entry in $(patch_markers "$1"); do
        file="${entry%%:*}"
        needle="${entry#*:}"
        if [ ! -f "$VP_DIR/$file" ] || ! grep -qF -- "$needle" "$VP_DIR/$file"; then
            return 1
        fi
    done
    return 0
}

apply_patches() {
    local pf path
    for pf in "${PATCH_FILES[@]}"; do
        path="$PATCH_DIR/$pf"
        [ -f "$path" ] || die "missing patch file: $path"
        if git -C "$VP_DIR" apply --check "$path" 2>/dev/null; then
            git -C "$VP_DIR" apply "$path"
            log "  applied patch $pf"
        elif patch_applied "$pf"; then
            log "  skip patch $pf; already applied"
        else
            die "cannot apply $pf to $VP_DIR; upstream moved?"
        fi
    done
}

build_videopipe() {
    # build_videopipe ON|OFF: configure and build. Return non-zero on failure.
    local llm="$1"
    rm -rf "$VP_BUILD_DIR"
    mkdir -p "$VP_BUILD_DIR"
    run_logged "$LOG_DIR/videopipe-cmake-llm-$llm.log" \
        cmake -S "$VP_DIR" -B "$VP_BUILD_DIR" \
            -DCMAKE_PREFIX_PATH="$OCV_INSTALL_DIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DVP_WITH_CUDA=OFF \
            -DVP_WITH_TRT=OFF \
            -DVP_WITH_PADDLE=OFF \
            -DVP_WITH_KAFKA=OFF \
            -DVP_WITH_GSTREAMER=$VP_GST \
            -DVP_BUILD_SAMPLES=$SAMPLES \
            -DVP_WITH_LLM=$llm \
        || return 1
    run_logged "$LOG_DIR/videopipe-make-llm-$llm.log" \
        make -C "$VP_BUILD_DIR" -j"$JOBS" || return 1
}

stage_videopipe() {
    log "STAGE 3/4 videopipe"
    if [ ! -d "$VP_DIR/.git" ]; then
        log "  clone $VP_REPO"
        run_logged "$LOG_DIR/videopipe-clone.log" \
            git clone --depth 1 "$VP_REPO" "$VP_DIR"
    else
        log "  skip clone; $VP_DIR exists"
    fi
    VIDEOPIPE_COMMIT="$(git -C "$VP_DIR" rev-parse HEAD)"

    apply_patches

    # Stamp key covers commit, flags, and patch content. Any change rebuilds.
    local patch_hash pf so
    patch_hash=""
    for pf in "${PATCH_FILES[@]}"; do
        patch_hash="$patch_hash$(md5sum "$PATCH_DIR/$pf" | cut -d' ' -f1)"
    done
    patch_hash="$(printf '%s' "$patch_hash" | md5sum | cut -d' ' -f1)"
    so="$VP_BUILD_DIR/libs/libvideo_pipe.so"
    if [ -f "$so" ] && [ -f "$STAMP_FILE" ] \
        && [ "$(cat "$STAMP_FILE")" = "$VIDEOPIPE_COMMIT:$EXPECTED_LLM:$VP_GST:$SAMPLES:$patch_hash" ]; then
        log "  skip build; $so matches stamp $(cat "$STAMP_FILE")"
        return 0
    fi

    log "  build CPU-only with $JOBS jobs (try VP_WITH_LLM=$EXPECTED_LLM first)"
    if [ "$EXPECTED_LLM" = "ON" ]; then
        if ! build_videopipe ON; then
            warn "VP_WITH_LLM=ON failed; retry with OFF and record the fallback"
            EXPECTED_LLM="OFF"
            LLM_FALLBACK="yes"
            build_videopipe OFF || die "videopipe build failed twice; see $LOG_DIR"
        fi
    else
        build_videopipe OFF || die "videopipe build failed; see $LOG_DIR"
    fi
    printf '%s\n' "$VIDEOPIPE_COMMIT:$EXPECTED_LLM:$VP_GST:$SAMPLES:$patch_hash" >"$STAMP_FILE"
    log "  built $so"
}

# ---------------------------------------------------------------------------
# Stage 4: verify
# ---------------------------------------------------------------------------
stage_verify() {
    log "STAGE 4/4 verify"
    local so="$VP_BUILD_DIR/libs/libvideo_pipe.so"
    [ -f "$so" ] || die "missing artifact: $so"

    export LD_LIBRARY_PATH="$OCV_INSTALL_DIR/lib:$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ldd "$so" >"$LOG_DIR/videopipe-ldd.txt" 2>&1 || true
    if grep -q "not found" "$LOG_DIR/videopipe-ldd.txt"; then
        grep "not found" "$LOG_DIR/videopipe-ldd.txt" >&2 || true
        die "unresolved libraries; see $LOG_DIR/videopipe-ldd.txt"
    fi
    local n
    n="$(grep -c '=>' "$LOG_DIR/videopipe-ldd.txt" || true)"
    log "  all libraries resolve ($n dynamic entries)"

    # LLM symbols must exist in the library when VP_WITH_LLM=ON.
    if [ "$EXPECTED_LLM" = "ON" ]; then
        local nllm
        nllm="$(nm -D "$so" | grep -c mllm || true)"
        [ "${nllm:-0}" -gt 0 ] || die "no mllm symbols in libvideo_pipe.so"
        log "  llm symbols present ($nllm dynamic symbols match mllm)"
    fi

    # Decode test. It proves FFmpeg decode works in the vendored OpenCV.
    local test_bin="$THIRD/tests/build/test_ffmpeg_decode"
    mkdir -p "$(dirname "$test_bin")"
    run_logged "$LOG_DIR/decode-test-build.log" \
        g++ -std=c++17 -O2 -I"$OCV_INSTALL_DIR/include/opencv4" \
            -o "$test_bin" "$THIRD/tests/test_ffmpeg_decode.cpp" \
            -L"$OCV_INSTALL_DIR/lib" -lopencv_videoio -lopencv_core \
        || die "decode test compile failed; see $LOG_DIR/decode-test-build.log"
    if [ -f "$DECODE_TEST_VIDEO" ]; then
        run_logged "$LOG_DIR/decode-test-run.log" "$test_bin" "$DECODE_TEST_VIDEO" \
            || die "decode test failed; see $LOG_DIR/decode-test-run.log"
        log "  decode test: $(grep 'decoded frames' "$LOG_DIR/decode-test-run.log")"
    else
        warn "decode test video missing: $DECODE_TEST_VIDEO (set TBA_DECODE_TEST_VIDEO)"
    fi
}

write_versions_env() {
    local zoo_flag=0
    [ "$FFMPEG_SOURCE" = "zoo" ] && zoo_flag=1
    # Quote every value. env.sh sources this file, so values with spaces or
    # semicolons must stay quoted.
    local patches_list
    patches_list="${PATCH_FILES[*]}"
    cat >"$VERSIONS_ENV" <<EOF
# Generated by tools/trashbot_annotator/setup_env.sh. Do not edit by hand.
TBA_SETUP_DATE="$(date +%Y-%m-%dT%H:%M:%S%z)"
FFMPEG_SOURCE="$FFMPEG_SOURCE"
USE_ZOO_FFMPEG=$zoo_flag
FFMPEG_VERSION="$FFMPEG_LIB_VERSION"
FFMPEG_ROOT="$FFMPEG_ROOT"
FFMPEG_PKG_CONFIG="$FFMPEG_PC_DIR"
OPENCV_VERSION="$OPENCV_VER"
OPENCV_ROOT="$OCV_INSTALL_DIR"
OPENCV_BUILD_LIST="core,imgproc,imgcodecs,videoio,highgui,video,dnn,freetype"
OPENCV_FLAGS="WITH_FFMPEG=ON WITH_GSTREAMER=OFF WITH_GTK=OFF WITH_QT=OFF BUILD_SHARED_LIBS=ON"
OPENCV_BUILD_LIST_NOTE="dnn from main tree; freetype from contrib; host FreeType and HarfBuzz"
VIDEOPIPE_REPO="$VP_REPO"
VIDEOPIPE_COMMIT="$VIDEOPIPE_COMMIT"
VIDEOPIPE_PATCHES="$patches_list"
VP_WITH_CUDA=OFF
VP_WITH_TRT=OFF
VP_WITH_PADDLE=OFF
VP_WITH_KAFKA=OFF
VP_WITH_GSTREAMER=$VP_GST
VP_BUILD_SAMPLES=$SAMPLES
VP_WITH_LLM="$EXPECTED_LLM"
VP_LLM_FALLBACK="$LLM_FALLBACK"
EOF
    log "wrote $VERSIONS_ENV"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
require_tools

FFMPEG_ROOT="$ZOO_FFMPEG"
FFMPEG_SOURCE="unknown"
FFMPEG_PC_DIR=""
FFMPEG_LIB_VERSION="unknown"
VIDEOPIPE_COMMIT="unknown"
EXPECTED_LLM="${TBA_VP_WITH_LLM:-ON}"
LLM_FALLBACK="no"
STAMP_FILE="$THIRD/.stamps/videopipe-build"

mkdir -p "$(dirname "$STAMP_FILE")"

stages=("$@")
if [ "${#stages[@]}" -eq 0 ]; then
    stages=(ffmpeg opencv videopipe verify)
fi

for s in "${stages[@]}"; do
    case "$s" in
        ffmpeg)    stage_ffmpeg ;;
        opencv)    stage_opencv ;;
        videopipe) stage_videopipe ;;
        verify)    stage_verify ;;
        *) die "unknown stage: $s (use: ffmpeg opencv videopipe verify)" ;;
    esac
done

write_versions_env
log "done"
