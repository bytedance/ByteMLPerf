project(vector_peak_performance)
cmake_minimum_required(VERSION 3.15)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fPIC -std=c++11 -pthread -pipe")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -fPIC")

################################################################################
# Neuware Evironment and BANG Setup
################################################################################

# check `NEUWARE_HOME` env
message(${NEUWARE_HOME})
if(EXISTS ${NEUWARE_HOME})
  include_directories("${NEUWARE_HOME}/include")
  link_directories("${NEUWARE_HOME}/lib64")
  link_directories("${NEUWARE_HOME}/lib")
  set(NEUWARE_ROOT_DIR "${NEUWARE_HOME}")
else()
  message(FATAL_ERROR "NEUWARE directory cannot be found, refer README.md to prepare NEUWARE_HOME environment.")
endif()

# setup cmake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${CMAKE_SOURCE_DIR}/cmake"
  "${NEUWARE_HOME}/cmake"
  "${NEUWARE_HOME}/cmake/modules"
)

# include FindBANG.cmake and check cncc
find_package(BANG)
if(NOT BANG_FOUND)
  message(FATAL_ERROR "BANG cannot be found.")
elseif (NOT BANG_CNCC_EXECUTABLE)
  message(FATAL_ERROR "cncc not found, please ensure cncc is in your PATH env or set variable BANG_CNCC_EXECUTABLE from cmake. Otherwise you should check path used by find_program(BANG_CNCC_EXECUTABLE) in FindBANG.cmake")
endif()
message(STATUS "BANG_CNCC_EXECUTABLE=${BANG_CNCC_EXECUTABLE}")
execute_process(
  COMMAND ${BANG_CNCC_EXECUTABLE} --version
  COMMAND head -n1
  COMMAND awk "{print $2}"
  COMMAND sed "s/^v//g"
  OUTPUT_VARIABLE _cncc_version
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "cncc version ${_cncc_version}")
if(NOT "${_cncc_version}" VERSION_LESS "3.8.1")
#  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -fno-conv-warmup")
else()
  if(NOT "${_cncc_version}" VERSION_LESS "3.7.3")
    message(FATAL_ERROR "cncc version must less than 3.7.3 or higher than 3.8.1.")
  endif()
endif()

# setup cncc flags
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -fPIC -Wall -std=c++11 -pthread")
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3")
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_372"
                                         "--bang-mlu-arch=mtp_592"
                                         "--bang-mlu-arch=mtp_613"
                                         "--bang-wram-align64"
                                         "--no-neuware-version-check"
)

include_directories(tests "${CMAKE_CURRENT_SOURCE_DIR}/kernels")

file(GLOB_RECURSE src_files ${src_files} "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.mlu")
set(src_files ${src_files} "${CMAKE_CURRENT_SOURCE_DIR}/main.mlu")
bang_add_executable(tests ${src_files})
target_link_libraries(tests cnrt cndrv)
