#!/usr/bin/env bash
#
# env.sh - export the trashbot annotator native environment.
#
# Usage: source tools/trashbot_annotator/env.sh
# After this, pkg-config, cmake and the dynamic linker resolve the vendored
# OpenCV, FFmpeg and VideoPipe trees.

_TBA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_TBA_THIRD="$_TBA_DIR/third_party"

export TRASHBOT_THIRD_PARTY="$_TBA_THIRD"
export OPENCV_ROOT="$_TBA_THIRD/opencv/install"
export VIDEOPIPE_ROOT="$_TBA_THIRD/VideoPipe"

# versions.env records the FFmpeg choice made by setup_env.sh. Source it when
# present; fall back to detection otherwise.
if [ -f "$_TBA_THIRD/versions.env" ]; then
    # shellcheck disable=SC1091
    . "$_TBA_THIRD/versions.env"
fi
if [ -z "${FFMPEG_ROOT:-}" ]; then
    if [ -f "$_TBA_THIRD/ffmpeg/install/lib/pkgconfig/libavcodec.pc" ]; then
        FFMPEG_ROOT="$_TBA_THIRD/ffmpeg/install"
    else
        FFMPEG_ROOT="/share/pierson/matt/zoo/ffmpeg-libs/n7.1"
    fi
    export FFMPEG_ROOT
fi

# tba_prepend VAR DIR: add DIR to VAR when DIR exists and is not in VAR yet.
tba_prepend() {
    local var="$1" dir="$2" cur
    [ -d "$dir" ] || return 0
    cur="${!var:-}"
    case ":$cur:" in
        *":$dir:"*) return 0 ;;
    esac
    if [ -n "$cur" ]; then
        export "$var=$dir:$cur"
    else
        export "$var=$dir"
    fi
}

tba_prepend PATH "$OPENCV_ROOT/bin"
tba_prepend PKG_CONFIG_PATH "$OPENCV_ROOT/lib/pkgconfig"
tba_prepend PKG_CONFIG_PATH "$FFMPEG_ROOT/lib/pkgconfig"
tba_prepend LD_LIBRARY_PATH "$OPENCV_ROOT/lib"
tba_prepend LD_LIBRARY_PATH "$FFMPEG_ROOT/lib"
tba_prepend CMAKE_PREFIX_PATH "$OPENCV_ROOT"
tba_prepend CMAKE_PREFIX_PATH "$FFMPEG_ROOT"

unset _TBA_DIR _TBA_THIRD tba_prepend
