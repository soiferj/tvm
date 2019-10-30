#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>

namespace tvm {
namespace contrib {

using namespace runtime;

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.tensorflow.run")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    std::cout << "In TF native op" << std::endl;
    std::string input1 = args[3];
    std::string input2 = args[4];
    std::string output = args[5];
    std::cout << "input names: " << input1 << "," << input2 << std::endl;
    std::cout << "output name: " << output << std::endl;
});

}  // namespace contrib
}  // namespace tvm