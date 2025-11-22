cmake_minimum_required(VERSION 3.15)
project(whisper_benchmark_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime - try CMake package first, then manual detection
find_package(ONNX QUIET)

# Find required packages
find_package(sndfile QUIET)
find_package(jsoncpp QUIET)

# Fallback to manual detection if CMake package not found
if(NOT ONNX_FOUND)
    # Try to find ONNX Runtime manually (especially in /opt/onnxruntime from Docker)
    find_library(ONNX_LIB NAMES onnxruntime PATHS /opt/onnxruntime/lib NO_DEFAULT_PATH)
    find_library(ONNX_LIB NAMES onnxruntime)
    find_path(ONNX_INCLUDE_DIR NAMES onnxruntime_cxx_api.h PATHS /opt/onnxruntime/include NO_DEFAULT_PATH)
    find_path(ONNX_INCLUDE_DIR NAMES onnxruntime_cxx_api.h)
    if(NOT ONNX_LIB OR NOT ONNX_INCLUDE_DIR)
        message(FATAL_ERROR "ONNX Runtime not found. Install it or set ONNX paths manually")
    endif()
    # Add all ONNX Runtime include subdirectories
    set(ONNX_INCLUDE_DIRS 
        ${ONNX_INCLUDE_DIR}
        ${ONNX_INCLUDE_DIR}/onnxruntime
        ${ONNX_INCLUDE_DIR}/onnxruntime/core/session
    )
    message(STATUS "ONNX Runtime found manually: ${ONNX_LIB}")
    message(STATUS "ONNX Runtime include: ${ONNX_INCLUDE_DIR}")
else()
    set(ONNX_INCLUDE_DIRS ${ONNX_INCLUDE_DIR})
endif()

if(NOT SNDFILE_FOUND)
    find_library(SNDFILE_LIB NAMES sndfile)
    find_path(SNDFILE_INCLUDE_DIR NAMES sndfile.h)
    if(SNDFILE_LIB AND SNDFILE_INCLUDE_DIR)
        set(SNDFILE_LIBRARIES ${SNDFILE_LIB})
        set(SNDFILE_INCLUDE_DIRS ${SNDFILE_INCLUDE_DIR})
        set(HAVE_SNDFILE ON)
        message(STATUS "libsndfile found: ${SNDFILE_LIB}")
    else()
        message(WARNING "libsndfile not found. Audio file support will be limited (WAV only).")
        set(HAVE_SNDFILE OFF)
    endif()
else()
    set(HAVE_SNDFILE ON)
    message(STATUS "libsndfile found via CMake package")
endif()

if(NOT JSONCPP_FOUND)
    find_library(JSONCPP_LIB NAMES jsoncpp PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
    find_path(JSONCPP_INCLUDE_DIR NAMES json/json.h PATHS /usr/include/jsoncpp /usr/include /usr/local/include)
    if(JSONCPP_LIB AND JSONCPP_INCLUDE_DIR)
        set(JSONCPP_LIBRARIES ${JSONCPP_LIB})
        set(JSONCPP_INCLUDE_DIRS ${JSONCPP_INCLUDE_DIR})
        message(STATUS "jsoncpp found manually: ${JSONCPP_LIB}")
    else()
        message(WARNING "jsoncpp not found. Using fallback JSON handling.")
        set(JSONCPP_INCLUDE_DIRS "")
    endif()
else()
    message(STATUS "jsoncpp found via CMake package")
endif()

# Create include list
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNX_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/onnx_model.cpp
    src/audio_utils.cpp
    src/metrics.cpp
)

set(HEADERS
    include/onnx_model.hpp
    include/audio_utils.hpp
    include/metrics.hpp
)

# Create executable
add_executable(whisper_benchmark ${SOURCES} ${HEADERS})

# Link libraries
if(ONNX_FOUND)
    target_link_libraries(whisper_benchmark PRIVATE ONNX::ONNX)
else()
    target_link_libraries(whisper_benchmark PRIVATE ${ONNX_LIB})
endif()

if(SNDFILE_FOUND)
    target_link_libraries(whisper_benchmark PRIVATE sndfile::sndfile)
elseif(SNDFILE_LIB)
    target_link_libraries(whisper_benchmark PRIVATE ${SNDFILE_LIB})
endif()

if(JSONCPP_FOUND)
    target_link_libraries(whisper_benchmark PRIVATE jsoncpp_lib)
elseif(JSONCPP_LIB)
    target_link_libraries(whisper_benchmark PRIVATE ${JSONCPP_LIB})
endif()

# Include directories
target_include_directories(whisper_benchmark PRIVATE ${INCLUDE_DIRS})

# Add preprocessor define for SNDFILE
if(HAVE_SNDFILE)
    target_compile_definitions(whisper_benchmark PRIVATE HAVE_SNDFILE)
endif()

# Platform-specific settings
if(MSVC)
    target_compile_options(whisper_benchmark PRIVATE /W4 /O2)
else()
    target_compile_options(whisper_benchmark PRIVATE -Wall -Wextra -O2)
endif()

# Installation
install(TARGETS whisper_benchmark DESTINATION bin)
