# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  mse_gain.hpp
  mad_gain.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

#add_cli_executable(decision_tree)
#add_python_binding(decision_tree)
#add_julia_binding(decision_tree)
#add_go_binding(decision_tree)
#add_r_binding(decision_tree)
#add_markdown_docs(decision_tree "cli;python;julia;go;r" "classification")
