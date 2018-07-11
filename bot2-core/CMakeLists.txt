cmake_minimum_required(VERSION 2.6.0)
# Doesn't throw error for get_target_property() on non existent target
cmake_policy(SET CMP0045 OLD)

# Doesn't throw error for get_target_property() when no LOCATION available
cmake_policy(SET CMP0026 OLD)

# Dipping into and out of other CMake files doesn't mess with currect config
cmake_policy(SET CMP0011 NEW)

set(POD_NAME bot2-core)

include(cmake/pods.cmake)

include(cmake/lcmtypes.cmake)
lcmtypes_build(C_AGGREGATE_HEADER bot_core.h CPP_AGGREGATE_HEADER bot_core.hpp)

add_subdirectory(src/bot_core)
add_subdirectory(java)
