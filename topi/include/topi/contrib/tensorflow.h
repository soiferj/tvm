/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief External function interface to cuBLAS libraries
 * \file cublas.h
 */
#ifndef TOPI_CONTRIB_TENSORFLOW_H_
#define TOPI_CONTRIB_TENSORFLOW_H_

#include "tvm/operation.h"
#include "topi/detail/extern.h"

namespace topi {
namespace contrib {
using namespace tvm;
using namespace topi::detail;

inline Tensor tensorflow_native(const Tensor& input1,
                                const Tensor& input2,
                                std::string input1name,
                                std::string input2name,
                                std::string outputname,
                                std::string graph_def_str,
                                const std::string name = "T_tensorflow_native",
                                const std::string tag = "tensorflow_native") {
  return make_extern(
    { { 2, 3 } }, { input1->dtype }, { input1, input2 },
    [&](Array<Buffer> ins, Array<Buffer> outs) {
      return call_packed({
        Expr("tvm.contrib.tensorflow.run"),
        pack_buffer(ins[0]),
        pack_buffer(ins[1]),
        pack_buffer(outs[0]),
        input1name,
        input2name,
        outputname,
        graph_def_str });
    }, "C", "", {})[0];
}

}  // namespace contrib
}  // namespace topi

#endif  // TOPI_CONTRIB_TENSORFLOW_H_
