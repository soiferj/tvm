file(GLOB TENSORFLOW_CONTRIB_SRC src/runtime/contrib/tensorflow/*.cc)
list(APPEND RUNTIME_SRCS ${TENSORFLOW_CONTRIB_SRC})

include_directories("/home/jonso/anaconda3/envs/tvm/lib/python3.7/site-packages/tensorflow/include")

find_library(TENSORFLOW_LIBRARY NAMES tensorflow_framework HINTS "/home/jonso/anaconda3/envs/tvm/lib/python3.7/site-packages/tensorflow")
message("Tensorflow library: " ${TENSORFLOW_LIBRARY})
list(APPEND TVM_RUNTIME_LINKER_LIBS "/home/jonso/anaconda3/envs/tvm/lib/python3.7/site-packages/tensorflow/libtensorflow_framework.so.1")