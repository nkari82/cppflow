#pragma once
#include <tensorflow/lite/c/c_api.h>


namespace cppflow
{
	namespace lite
	{
		class model
		{
		public:
			explicit model(const std::string& filename)
			{
				model_ = { TfLiteModelCreateFromFile(filename.c_str()), TfLiteModelDelete };

				// Build the interpreter
				TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
				TfLiteInterpreterOptionsSetNumThreads(options, 1);

				// Create the interpreter.
				interpreter_ = { TfLiteInterpreterCreate(model_.get(), options), TfLiteInterpreterDelete };
				if (interpreter_ == nullptr)
					throw std::logic_error("Failed to create interpreter");
			}

		private:
			std::shared_ptr<TfLiteModel> model_;
			std::shared_ptr<TfLiteInterpreter> interpreter_;
		};
	}
}

// 