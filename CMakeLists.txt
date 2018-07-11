cmake_minimum_required(VERSION 3.3.0)
SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE OFF)

# to avoid changing source code
set(CMAKE_DISABLE_IN_SOURCE_BUILD OFF)
set(CMAKE_DISABLE_SOURCE_CHANGES OFF)
project(libbot)

###############################################################################
# Set up policies
# Doesn't throw error for get_target_property() on non existent target
cmake_policy(SET CMP0045 OLD)

# Doesn't throw error for get_target_property() when no LOCATION available
cmake_policy(SET CMP0026 OLD)

# Dipping into and out of other CMake files doesn't mess with currect config
cmake_policy(SET CMP0011 NEW)

include(ExternalProject)

set(COMMON_CMAKE_ARGS "")

get_cmake_property(CACHE_VARS CACHE_VARIABLES)
foreach(CACHE_VAR ${CACHE_VARS})
  get_property(VAR_HELPSTRING CACHE ${CACHE_VAR} PROPERTY HELPSTRING)
  if(CACHE_VAR_HELPSTRING STREQUAL "No help, variable specified on the command line.")
    get_property(CACHE_VAR_TYPE CACHE ${CACHE_VAR} PROPERTY TYPE)
    if(CACHE_VAR_TYPE STREQUAL "UNINITIALIZED")
      set(CACHE_VAR_TYPE)
    else()
      set(CACHE_VAR_TYPE :${CACHE_VAR_TYPE})
    endif()
    MESSAGE("-D${CACHE_VAR}${CACHE_VAR_TYPE}=${${CACHE_VAR}}")
    list(APPEND COMMON_CMAKE_ARGS "-D${CACHE_VAR}${CACHE_VAR_TYPE}=${${CACHE_VAR}}")
  endif()
endforeach()

set(CMAKE_COPY_ARGS
  "CMAKE_INSTALL_PREFIX"
  "CMAKE_BUILD_TYPE"
  "CMAKE_SOURCE_DIR"
  "CMAKE_LIBRARY_OUTPUT_DIRECTORY"
  "CMAKE_ARCHIVE_OUTPUT_DIRECTORY"
  "CMAKE_RUNTIME_OUTPUT_DIRECTORY"
  )
foreach (carg ${CMAKE_COPY_ARGS})
  list(APPEND COMMON_CMAKE_ARGS "-D${carg}:STRING=${${carg}}")
endforeach()

ExternalProject_Add(bot2-core
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/bot2-core
        CMAKE_CACHE_ARGS ${COMMON_CMAKE_ARGS})

ExternalProject_Add(bot2-vis
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/bot2-vis
        CMAKE_ARGS ${COMMON_CMAKE_ARGS} bot2-core)

ExternalProject_Add(bot2-procman
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/bot2-procman
        CMAKE_ARGS ${COMMON_CMAKE_ARGS})

ExternalProject_Add(bot2-lcmgl
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/bot2-lcmgl
        CMAKE_ARGS ${COMMON_CMAKE_ARGS} bot2-vis)

ExternalProject_Add(bot2-lcm-utils
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/bot2-lcm-utils
        CMAKE_ARGS ${COMMON_CMAKE_ARGS})

ExternalProject_Add(bot2-param
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/bot2-param
        CMAKE_ARGS ${COMMON_CMAKE_ARGS} bot2-core)

ExternalProject_Add(bot2-frames
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/bot2-frames
        CMAKE_ARGS ${COMMON_CMAKE_ARGS}
DEPENDS bot2-core bot2-param bot2-vis)
