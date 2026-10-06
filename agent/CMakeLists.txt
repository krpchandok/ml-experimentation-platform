cmake_minimum_required(VERSION 3.16)
project(mlplat_agent LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(mlplat_agent_core STATIC
  src/procfs.cpp
  src/process_tree.cpp
  src/sampler.cpp
  src/json_writer.cpp
  src/records.cpp
  src/nvml.cpp
)
target_include_directories(mlplat_agent_core PUBLIC src)
target_compile_features(mlplat_agent_core PUBLIC cxx_std_17)
target_link_libraries(mlplat_agent_core PUBLIC ${CMAKE_DL_LIBS})
set_target_properties(mlplat_agent_core PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(mlplat_agent_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(mlplat-agent src/main.cpp)
target_link_libraries(mlplat-agent PRIVATE mlplat_agent_core)
set_target_properties(mlplat-agent PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(mlplat-agent PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
  add_library(fake_nvml MODULE tests/fake_nvml.cpp)
  set_target_properties(fake_nvml PROPERTIES PREFIX "lib" OUTPUT_NAME "fake-nvidia-ml")
  add_executable(mlplat_agent_tests
    tests/test_main.cpp
    tests/test_procfs.cpp
    tests/test_process_tree.cpp
    tests/test_sampler.cpp
    tests/test_json.cpp
    tests/test_nvml.cpp
  )
  target_link_libraries(mlplat_agent_tests PRIVATE mlplat_agent_core)
  add_dependencies(mlplat_agent_tests fake_nvml)
  target_compile_features(mlplat_agent_tests PRIVATE cxx_std_20)
  target_compile_definitions(mlplat_agent_tests PRIVATE
    MLPLAT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    MLPLAT_FAKE_NVML="$<TARGET_FILE:fake_nvml>")
  target_compile_options(mlplat_agent_tests PRIVATE -Wall -Wextra)
  add_test(NAME agent_unit_tests COMMAND mlplat_agent_tests)
endif()
