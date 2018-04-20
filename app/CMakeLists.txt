# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/main.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Add dependent libraries
option(NewIPOPT "" ON)

if(NewIPOPT)

    add_library(blas SHARED IMPORTED)
    set_property(TARGET blas PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libcoinblas.so)

    add_library(lapack SHARED IMPORTED)
    set_property(TARGET lapack PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libcoinlapack.so)

    add_library(metis SHARED IMPORTED)
    set_property(TARGET metis PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libcoinmetis.so)

    add_library(mumps SHARED IMPORTED)
    set_property(TARGET mumps PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libcoinmumps.so)

    add_library(ipopt SHARED IMPORTED)
    set_property(TARGET ipopt PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libipopt.so)

else()

    add_library(ipopt SHARED IMPORTED)
    set_property(TARGET ipopt PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libipoptmohan.so)

endif()


# Location of header files
include_directories(${CMAKE_SOURCE_DIR}/libs/include
                    ${CMAKE_SOURCE_DIR}/libs/include/coin
                    ${CMAKE_SOURCE_DIR}/libs/include/coin/ThirdParty)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

if(NewIPOPT)

    target_link_libraries(native-lib ${log-lib} blas lapack metis mumps ipopt)

else()

    target_link_libraries(native-lib ${log-lib} ipopt)

endif()
