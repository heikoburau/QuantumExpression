cmake_minimum_required(VERSION 3.8)
project(QuantumExpression LANGUAGES CXX)

# set helper pathes to find libraries and packages
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

# Project paths
# =============

set(project_include_dir "${CMAKE_CURRENT_LIST_DIR}/include")

# Installation
# ============

install(
	DIRECTORY "${project_include_dir}/"
	DESTINATION include/QuantumExpression
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

