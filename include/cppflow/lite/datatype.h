#pragma once

namespace cppflow
{
	namespace lite
	{
		using datatype = TfLiteType;

        /**
		 *
		 * @tparam T
		 * @return The TensorFlow type of T
		 */
        template<typename T>
        TfLiteType deduce_tf_type() {
            if (std::is_same<T, float>::value)
                return kTfLiteFloat32;
            if (std::is_same<T, double>::value)
                return kTfLiteFloat64;
            if (std::is_same<T, int32_t >::value)
                return kTfLiteInt32;
            if (std::is_same<T, uint8_t>::value)
                return kTfLiteUInt8;
            if (std::is_same<T, int16_t>::value)
                return kTfLiteInt16;
            if (std::is_same<T, int8_t>::value)
                return kTfLiteInt8;
            if (std::is_same<T, int64_t>::value)
                return kTfLiteInt64;
            if (std::is_same<T, unsigned char>::value)
                return kTfLiteBool;
            if (std::is_same<T, int16_t>::value)
                return kTfLiteInt16;

            throw std::runtime_error{ "Could not deduce type!" };
        }
	}
}