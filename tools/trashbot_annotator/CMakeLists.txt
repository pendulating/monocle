# TrashBot deployment footage annotator.
# Pipeline: video file -> batch VLM analyser -> mp4 writer or fake sink.
cmake_minimum_required(VERSION 3.10)
project(trashbot_annotator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV lives in third_party/opencv/install. The caller sets CMAKE_PREFIX_PATH.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(Threads REQUIRED)

# Locate VideoPipe. Accept two layouts:
# 1. tools/trashbot_annotator/third_party/VideoPipe
# 2. <repo root>/third_party/VideoPipe
set(VIDEOPIPE_DIR "")
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/VideoPipe/CMakeLists.txt)
    set(VIDEOPIPE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/VideoPipe)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/VideoPipe/CMakeLists.txt)
    get_filename_component(VIDEOPIPE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/VideoPipe" ABSOLUTE)
endif()

if(NOT VIDEOPIPE_DIR STREQUAL "")
    # VideoPipe reads version info from ${CMAKE_SOURCE_DIR}/utils/vp_version.h.in.
    # When we act as the top project, that path points into our source tree.
    # Provide a stub with the expected macros when the file is missing.
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
            AND NOT EXISTS ${CMAKE_SOURCE_DIR}/utils/vp_version.h.in)
        file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/utils)
        file(WRITE ${CMAKE_SOURCE_DIR}/utils/vp_version.h.in
"#pragma once\n\n#define APP_BUILD_TIME \"@BUILD_TIME@\"\n#define APP_GIT_COMMIT \"@FINAL_GIT_VERSION@\"\n#define APP_VERSION \"v@BUILD_TIME@-@FINAL_GIT_VERSION@\"\n")
    endif()

    # Honor the VP_WITH_* variables below. Without this policy, option()
    # in the VideoPipe file clears them and flips VP_WITH_LLM to OFF.
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    # Disable heavy extras. Keep the LLM part, we need cpp_llmlib headers.
    set(VP_WITH_CUDA OFF)
    set(VP_WITH_TRT OFF)
    set(VP_WITH_PADDLE OFF)
    set(VP_WITH_KAFKA OFF)
    set(VP_WITH_LLM ON)
    add_subdirectory(${VIDEOPIPE_DIR} ${CMAKE_BINARY_DIR}/video_pipe_build)
else()
    message(WARNING
        "trashbot_annotator: VideoPipe not found under third_party/. "
        "Configure succeeds, but target trashbot_pipe will not compile.")
endif()

add_executable(trashbot_pipe
    src/main.cpp
    src/vp_frame_batch_analyser_node.cpp
    src/vllm_client.cpp)

target_include_directories(trashbot_pipe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vendor
    ${OpenCV_INCLUDE_DIRS})

if(NOT VIDEOPIPE_DIR STREQUAL "")
    target_include_directories(trashbot_pipe PRIVATE
        ${VIDEOPIPE_DIR}/nodes
        ${VIDEOPIPE_DIR}/objects
        ${VIDEOPIPE_DIR}/utils
        # Header-only LLM client. No library target exists for it.
        ${VIDEOPIPE_DIR}/third_party/cpp_llmlib)
endif()

if(TARGET video_pipe)
    target_link_libraries(trashbot_pipe PRIVATE video_pipe ${OpenCV_LIBS} Threads::Threads)
else()
    target_link_libraries(trashbot_pipe PRIVATE ${OpenCV_LIBS} Threads::Threads)
endif()

# std::filesystem needs an extra lib on GCC 8 and older.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(trashbot_pipe PRIVATE stdc++fs)
endif()
