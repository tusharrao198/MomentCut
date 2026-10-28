cmake_minimum_required(VERSION 3.18)
project(openshorts_ced_events LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CED_CPP_SOURCE_DIR "" CACHE PATH "Path to the pinned ced.cpp checkout")
set(OPENSHORTS_CED_CPP_REVISION "" CACHE STRING "Pinned ced.cpp revision")

if(NOT APPLE)
  message(FATAL_ERROR "openshorts-ced-events is a native macOS worker")
endif()
if(NOT IS_ABSOLUTE "${CED_CPP_SOURCE_DIR}")
  message(FATAL_ERROR "CED_CPP_SOURCE_DIR must be an absolute ced.cpp checkout path")
endif()
if(NOT OPENSHORTS_CED_CPP_REVISION)
  message(FATAL_ERROR "OPENSHORTS_CED_CPP_REVISION is required")
endif()
foreach(required_file CMakeLists.txt src/ced.hpp src/audio_io.hpp)
  if(NOT EXISTS "${CED_CPP_SOURCE_DIR}/${required_file}")
    message(FATAL_ERROR "CED_CPP_SOURCE_DIR is missing ${required_file}")
  endif()
endforeach()

find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${CED_CPP_SOURCE_DIR}" status --porcelain
  RESULT_VARIABLE ced_git_status_result
  OUTPUT_VARIABLE ced_git_status
  ERROR_VARIABLE ced_git_status_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ced_git_status_result EQUAL 0 OR NOT "${ced_git_status}" STREQUAL "")
  message(FATAL_ERROR "CED_CPP_SOURCE_DIR must be a clean Git checkout")
endif()
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${CED_CPP_SOURCE_DIR}" rev-parse HEAD
  RESULT_VARIABLE ced_git_revision_result
  OUTPUT_VARIABLE ced_git_revision
  ERROR_VARIABLE ced_git_revision_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ced_git_revision_result EQUAL 0)
  message(FATAL_ERROR "Cannot resolve the ced.cpp checkout revision")
endif()
if(NOT "${ced_git_revision}" STREQUAL "${OPENSHORTS_CED_CPP_REVISION}")
  message(FATAL_ERROR
    "CED_CPP_SOURCE_DIR revision ${ced_git_revision} does not match "
    "OPENSHORTS_CED_CPP_REVISION ${OPENSHORTS_CED_CPP_REVISION}")
endif()

set(CED_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(CED_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CED_GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
add_subdirectory("${CED_CPP_SOURCE_DIR}" ced.cpp)

add_executable(openshorts-ced-events main.cpp)
target_include_directories(openshorts-ced-events PRIVATE "${CED_CPP_SOURCE_DIR}/src")
target_compile_definitions(openshorts-ced-events PRIVATE
  OPENSHORTS_CED_CPP_REVISION="${OPENSHORTS_CED_CPP_REVISION}"
  OPENSHORTS_CED_METAL=1)
target_link_libraries(openshorts-ced-events PRIVATE ced)
