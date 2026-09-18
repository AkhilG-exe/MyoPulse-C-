cmake_minimum_required(VERSION 3.16)

project(intent_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(intent_engine STATIC
    src/emg_filter.cpp
    src/feature_extractor.cpp
    src/intent_classifier.cpp
    src/proportional_control.cpp
)

target_include_directories(intent_engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(intent_engine PRIVATE /W4)
else()
    target_compile_options(intent_engine PRIVATE -Wall -Wextra -pedantic)
endif()

include(GNUInstallDirs)

install(TARGETS intent_engine
    EXPORT intent_engineTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/intent_engine DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT intent_engineTargets
    NAMESPACE intent_engine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/intent_engine
)