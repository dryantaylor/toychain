cmake_minimum_required(VERSION 3.16)

project(toychain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    set(OpenSSL_ROOT "C:/msys64/ucrt64")
    set(OPENSSL_ROOT_DIR "C:/msys64/ucrt64")
    list(APPEND CMAKE_PREFIX_PATH "C:/msys64/ucrt64")
endif()

find_package(OpenSSL REQUIRED)
find_package(OpenCL REQUIRED)

file(GLOB_RECURSE PROJECT_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)


add_executable(toychain ${PROJECT_SOURCES})
target_include_directories(toychain PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(toychain PRIVATE OpenSSL::Crypto OpenCL::OpenCL)
