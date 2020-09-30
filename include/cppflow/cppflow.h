//
// Created by serizba on 17/9/20.
//

#ifndef CPPFLOW_H
#define CPPFLOW_H
#include <tensorflow/c/c_api.h>
#include "tensor.h"
#include "model.h"
#include "raw_ops.h"
#include "ops.h"
#include "datatype.h"

namespace cppflow {

    /**
     * Version of TensorFlow and CppFlow
     * @return A string containing the version of TensorFow and CppFlow
     */
    std::string version();

}

/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/

namespace cppflow {
    std::string version() { return "TensorFlow: " + std::string(TF_Version()) + " CppFlow: 2.0.0";}
}

#endif // CPPFLOW_H
