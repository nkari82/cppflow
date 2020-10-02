#pragma once
#include <tensorflow/lite/c/c_api.h>
#include "datatype.h"
#include "tensor.h"
#include "model.h"

namespace cppflow 
{
	namespace lite
	{
		std::string version() { return "TensorFlowLite: " + std::string(TfLiteVersion()) + " CppFlow: 2.0.0"; }
	}
}