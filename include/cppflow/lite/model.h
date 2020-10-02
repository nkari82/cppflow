#pragma once
//#include <tensorflow/lite/c/c_api.h>
//#include "tensor.h"

namespace cppflow
{
	namespace lite
	{
		class model
		{
		public:
			explicit model(const char* path, size_t size = 0)
			{
				model_ = 
				{ 
					(size != 0) 
					? TfLiteModelCreate(path, size) 
					: TfLiteModelCreateFromFile(path)
					, TfLiteModelDelete 
				};
				assert(model_.get() != nullptr);

				// Build the interpreter
				options_ = { TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete };
				TfLiteInterpreterOptionsSetErrorReporter(options_.get(), reporter, this);
				TfLiteInterpreterOptionsSetNumThreads(options_.get(), 1);

				// Create the interpreter.
				interpreter_ = { TfLiteInterpreterCreate(model_.get(), options_.get()), TfLiteInterpreterDelete };
				if (interpreter_ == nullptr)
					throw std::logic_error("Failed to create interpreter");

				// Allocate tensor buffers.
				if (TfLiteInterpreterAllocateTensors(interpreter_.get()) != kTfLiteOk)
					throw std::logic_error("Failed to allocate tensors!");

				// Find input tensors.
				int32_t input_count = TfLiteInterpreterGetInputTensorCount(interpreter_.get());
				if(!input_count)
					throw std::logic_error("no input!");

				// Find output tensors.
				int32_t output_count = TfLiteInterpreterGetOutputTensorCount(interpreter_.get());
				if (!output_count)
					throw std::logic_error("no output!");

				for (int32_t i = 0; i < input_count; ++i)
				{
					TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(interpreter_.get(), i);
					inputs_.emplace_back(tensor);
				}
					
				for (int32_t i = 0; i < output_count; ++i)
				{
					const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(interpreter_.get(), i);
					outputs_.emplace_back(tensor);
				}
			}

			tensor operator()(const std::vector<tensor>& inputs)
			{
				assert(inputs.size() == inputs_.size());
				for (size_t i = 0; i < inputs.size(); ++i)
				{
					const auto& shape = inputs[i].shape();
					const auto* dims = inputs_[i]->dims;
					assert(dims->size == shape.size());
					
					// check realloc
					bool _realloc(false);
					for (size_t k = 0; k < dims->size; ++k)
					{
						if (dims->data[k] != shape[k])
						{
							if( TfLiteInterpreterResizeInputTensor(interpreter_.get(), (int32_t)i, shape.data(), (int32_t)shape.size()) != kTfLiteOk )
								throw std::logic_error("Failed resize tensor!");

							_realloc = true;
							break;
						}
					}

					if(_realloc && TfLiteInterpreterAllocateTensors(interpreter_.get()) != kTfLiteOk)
						throw std::logic_error("Failed allocate tensor!");

					std::memcpy(inputs_[i]->data.data, inputs[i].data(), inputs[i].size());
				}

				if (TfLiteInterpreterInvoke(interpreter_.get()) != kTfLiteOk)
					throw std::logic_error("Failed invoke!");

				return outputs_[0];
			}

		private:
			static void reporter(void* user_data, const char* format, va_list args)
			{
				vfprintf(stderr, format, args);
			}

		private:
			std::shared_ptr<TfLiteModel> model_;
			std::shared_ptr<TfLiteInterpreter> interpreter_;
			std::shared_ptr<TfLiteInterpreterOptions> options_;
			std::vector<TfLiteTensor*> inputs_;
			std::vector<const TfLiteTensor*> outputs_;
		};
	}
}