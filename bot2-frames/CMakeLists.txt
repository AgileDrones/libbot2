cmake_minimum_required(VERSION 2.6.0)
# Doesn't throw error for get_target_property() on non existent target
cmake_policy(SET CMP0045 OLD)

# Doesn't throw error for get_target_property() when no LOCATION available
cmake_policy(SET CMP0026 OLD)

# Dipping into and out of other CMake files doesn't mess with currect config
cmake_policy(SET CMP0011 NEW)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME bot2-frames)

include(cmake/pods.cmake)

include(cmake/lcmtypes.cmake)
lcmtypes_build()

find_package(PkgConfig REQUIRED)
find_package(OpenGL REQUIRED)

list(APPEND OPENGL_LIBRARIES GL)
set(GLUT_CFLAGS "")
set(GLUT_LIBRARIES -lglut)

set(LIBJPEG_CFLAGS "")
set(LIBJPEG_LIBRARIES -ljpeg)

pkg_check_modules(GLIB2 REQUIRED glib-2.0)
pkg_check_modules(LCM REQUIRED lcm)
pkg_check_modules(BOT2_CORE REQUIRED bot2-core)
pkg_check_modules(BOT2_PARAM REQUIRED bot2-param-client)

add_subdirectory(src)
add_subdirectory(src/test)

pkg_check_modules(BOT2_VIS bot2-vis)
if(NOT BOT2_VIS_FOUND)
    message("bot2-vis not found.  Not building libbot2 renderer")
else()
    add_subdirectory(src/renderer)
endif()
