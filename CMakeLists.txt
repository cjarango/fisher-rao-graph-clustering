cmake_minimum_required(VERSION 3.16)

project(frgc
    VERSION 0.1.0
    DESCRIPTION "Fisher-Rao graph clustering on statistical manifolds"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FRGC_BUILD_TESTS      "Build tests"               ON)
option(FRGC_BUILD_EXAMPLES   "Build examples"            ON)
option(FRGC_ENABLE_OPENMP    "Enable OpenMP"             ON)
option(FRGC_ENABLE_WARNINGS  "Enable compiler warnings"  ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(FRGC_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(FRGC_ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4 /permissive-)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic)
    endif()
endif()

file(GLOB_RECURSE FRGC_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/core/*.cpp"
)

add_library(frgc_core STATIC ${FRGC_SOURCES})

target_include_directories(frgc_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

if(FRGC_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(frgc_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(FRGC_BUILD_EXAMPLES)
    add_executable(frgc_app "${CMAKE_SOURCE_DIR}/src/app/main.cpp")
    target_link_libraries(frgc_app PRIVATE frgc_core)
endif()

if(FRGC_BUILD_TESTS)
    enable_testing()

    file(GLOB_RECURSE FRGC_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/tests/*.cpp"
    )

    foreach(test_src ${FRGC_TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)

        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE frgc_core)

        if(FRGC_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
            target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()

        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()