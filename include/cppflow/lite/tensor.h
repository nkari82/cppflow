#pragma once


namespace cppflow
{
	namespace lite
	{
		class tensor
		{
		public:
			void* data() const { return buffer_; }
			size_t size() const { return size_; }
			datatype dtype() { return type_; }
			const char* name() { return name_; }
			const std::vector<int32_t>& shape() const { return shape_; }

			tensor(const TfLiteTensor* _tensor)
				: tensor_(_tensor) 
				, name_(TfLiteTensorName(_tensor))
				, type_(TfLiteTensorType(_tensor))
				, size_(TfLiteTensorByteSize(_tensor))
				, buffer_(TfLiteTensorData(_tensor))
			{}

			template<typename T>
			tensor(T* value, size_t count) 
				: tensor(deduce_tf_type<T>(), (void*)value, count * sizeof(T), {(int32_t)count})
			{}

			template<typename T>
			tensor(T* value, size_t count, const std::vector<int32_t>& shape)
				: tensor(deduce_tf_type<T>(), (void*)value, count * sizeof(T), shape)
			{}

		private:
			tensor(TfLiteType type, void* buffer, size_t size, const std::vector<int32_t>& shape)
				: shape_(shape)
				, type_(type)
				, buffer_(buffer)
				, size_(size)
				, name_(nullptr)
				, tensor_(nullptr)
			{}

		private:
			std::vector<int32_t> shape_;
			datatype type_;
			void* buffer_;
			size_t size_;
			const char* name_;
			const TfLiteTensor* tensor_;
		};
	}
}