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

# --- Dependencies -----------------------------------------------------------

find_package(Boost CONFIG REQUIRED)

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

# --- Core library -----------------------------------------------------------

file(GLOB_RECURSE FRGC_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/core/*.cpp"
)

add_library(frgc_core STATIC ${FRGC_SOURCES})

target_include_directories(frgc_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(frgc_core PUBLIC Boost::boost)

if(FRGC_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(frgc_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- Example apps -----------------------------------------------------------

if(FRGC_BUILD_EXAMPLES)
    add_executable(frgc_visualize "${CMAKE_SOURCE_DIR}/src/app/visualize.cpp")
    target_link_libraries(frgc_visualize PRIVATE frgc_core)

    add_executable(frgc_experiment "${CMAKE_SOURCE_DIR}/src/app/main.cpp")
    target_link_libraries(frgc_experiment PRIVATE frgc_core)
endif()

# --- Tests ------------------------------------------------------------------

if(FRGC_BUILD_TESTS)
    enable_testing()

    find_package(doctest REQUIRED)

    # Unit tests
    file(GLOB_RECURSE FRGC_UNIT_TESTS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/tests/unit/*.cpp"
    )
    foreach(test_src ${FRGC_UNIT_TESTS})
        get_filename_component(test_name ${test_src} NAME_WE)

        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE frgc_core doctest::doctest)

        if(FRGC_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
            target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()

        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES LABELS "unit")
    endforeach()

    # Integration tests
    file(GLOB_RECURSE FRGC_INTEGRATION_TESTS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/tests/integration/*.cpp"
    )
    foreach(test_src ${FRGC_INTEGRATION_TESTS})
        get_filename_component(test_name ${test_src} NAME_WE)

        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE frgc_core doctest::doctest)

        if(FRGC_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
            target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()

        # test_driver_e2e needs the driver binary path baked in at compile time.
        if(test_name STREQUAL "test_driver_e2e")
            add_dependencies(${test_name} frgc_experiment)
            target_compile_definitions(${test_name} PRIVATE
                FRGC_EXPERIMENT_BIN="$<TARGET_FILE:frgc_experiment>")
        endif()

        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES LABELS "integration")
    endforeach()

    # Diagnostic programs: built but not registered with ctest
    file(GLOB_RECURSE FRGC_DIAGNOSTIC_TESTS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/tests/diagnostic/*.cpp"
    )
    foreach(test_src ${FRGC_DIAGNOSTIC_TESTS})
        get_filename_component(test_name ${test_src} NAME_WE)

        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE frgc_core doctest::doctest)

        if(FRGC_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
            target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endforeach()
endif()