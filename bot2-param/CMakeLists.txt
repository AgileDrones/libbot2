cmake_minimum_required(VERSION 2.6.0)
# Doesn't throw error for get_target_property() on non existent target
cmake_policy(SET CMP0045 OLD)

# Doesn't throw error for get_target_property() when no LOCATION available
cmake_policy(SET CMP0026 OLD)

# Dipping into and out of other CMake files doesn't mess with currect config
cmake_policy(SET CMP0011 NEW)

set(POD_NAME bot2-param)

include(cmake/pods.cmake)
pods_config_search_paths()

# automatically build LCM types.  This also defines a number of CMake
# variables, see cmake/lcmtypes.cmake for details
include(cmake/lcmtypes.cmake)
lcmtypes_build()

add_subdirectory(src/param_client)
add_subdirectory(src/param_server)
add_subdirectory(src/param_tester)
