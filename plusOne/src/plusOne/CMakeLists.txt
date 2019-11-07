set(CPP_FILES
    game.cpp
    )

set(HEADER_FILES
    game.h
    )

add_library(plusOne SHARED
    ${CPP_FILES}
    ${HEADER_FILES}
    )

target_include_directories(plusOne PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    )
set_target_properties(plusOne PROPERTIES
    PUBLIC_HEADER "${HEADER_FILES}"
    CXX_STANDARD 14
    )

# ====================================================================
# plusOne_python

set(PYTHONFILES
    setup.py
    plusOne/__init__.py
    plusOne/game.py
    )

pybind11_add_module(plusOne_python
    libplusOne_python.cpp
    )

target_include_directories(plusOne_python PUBLIC
    ${CMAKE_SOURCE_DIR}/pybind11/include
    ${CMAKE_SOURCE_DIR}/src
    ${PYTHON_INCLUDE_DIRS}
    )

target_link_libraries(plusOne_python PUBLIC
    ${PYTHON_LIBRARY}
    Eigen3::Eigen
    plusOne
    )

set_target_properties(plusOne_python PROPERTIES
    CXX_STANDARD 14
    )
if(WIN32)
    set_target_properties(plusOne_python PROPERTIES
        PREFIX "lib"
        SUFFIX ".pyd"
        )
else()
    set_target_properties(plusOne_python PROPERTIES
        PREFIX "lib"
        )
endif()

# Copy package to binary directory. This works differently on windows and linux.
# Because on windows one binary directory can have multiple build configurations, using visual studio. 
if(NOT "${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    add_custom_target(copyPython ALL)
    if(WIN32)
        if("${CMAKE_GENERATOR}" STREQUAL "NMake Makefiles")
            set_target_properties(plusOne_python PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pysrc/plusOne/")
            set_target_properties(plusOne_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pysrc/plusOne/")
            set_target_properties(plusOne PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pysrc/plusOne/")
            set_target_properties(plusOne PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pysrc/plusOne/")
            add_custom_command(TARGET copyPython POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/pysrc/MANIFEST.in.windows" "${CMAKE_BINARY_DIR}/pysrc/MANIFEST.in")
            foreach(PYFILE IN LISTS PYTHONFILES)
                add_custom_command(TARGET copyPython POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/pysrc/${PYFILE}" "${CMAKE_BINARY_DIR}/pysrc/${PYFILE}")
            endforeach()
        else()
            set(BUILD_TYPES Release Debug RelWithDebInfo MinSizeRel)
            set(BOOST_BUILD_TYPES RELEASE DEBUG RELEASE RELEASE)    
            foreach(val RANGE 3)
                list(GET BUILD_TYPES ${val} BUILD_TYPE)
                list(GET BOOST_BUILD_TYPES ${val} BOOST_BUILD_TYPE)
                string(TOUPPER "${BUILD_TYPE}" UPPER_BUILD_TYPE)
                copyBinary("${BOOST_BUILD_TYPE}" "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/pysrc/plusOne/" additionalBinaryFiles)
                set_target_properties(plusOne_python PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${UPPER_BUILD_TYPE} "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/pysrc/plusOne/")
                set_target_properties(plusOne_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${UPPER_BUILD_TYPE} "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/pysrc/plusOne/")
                set_target_properties(plusOne PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${UPPER_BUILD_TYPE} "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/pysrc/plusOne/")
                set_target_properties(plusOne PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${UPPER_BUILD_TYPE} "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/pysrc/plusOne/")
                add_custom_command(TARGET copyPython POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/pysrc/MANIFEST.in.windows" "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/pysrc/MANIFEST.in")
                foreach(PYFILE IN LISTS PYTHONFILES)
                    add_custom_command(TARGET copyPython POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/pysrc/${PYFILE}" "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/pysrc/${PYFILE}")
                endforeach()
            endforeach()
        endif()

    else()
        set_target_properties(plusOne_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pysrc/plusOne/")
        add_custom_command(TARGET copyPython POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/pysrc/MANIFEST.in.linux" "${CMAKE_BINARY_DIR}/${BUILD_TYPE}/pysrc/MANIFEST.in")
        foreach(PYFILE IN LISTS PYTHONFILES)
            add_custom_command(TARGET copyPython POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/pysrc/${PYFILE}" "${CMAKE_BINARY_DIR}/pysrc/${PYFILE}")
        endforeach()

    endif()

    install(CODE "EXECUTE_PROCESS(WORKING_DIRECTORY \"${CMAKE_BINARY_DIR}/${BUILD_TYPE}/pysrc/\" COMMAND ${PYTHON_EXECUTABLE} setup.py install)")

endif()
