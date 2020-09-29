//
// Created by serizba on 27/6/20.
//

#ifndef CPPFLOW2_TENSOR_H
#define CPPFLOW2_TENSOR_H

#include <memory>
#include <vector>
#include <cstring>
#include <string>
#include <tensorflow/c/tf_tensor.h>
#include <tensorflow/c/eager/c_api.h>

#include "context.h"
#include "datatype.h"

namespace cppflow {

    /**
     * @class tensor
     * @brief A TensorFlow eager tensor wrapper
     *
     */
    class tensor {
    public:
        tensor()= default;

        /**
        * Creates a tensor with the given values and specified shape
        * @tparam T A type that can be convertible into a tensor
        * @param values The values to be converted (in a flattened version)
        * @param shape The shape of the converted tensor
        */
        template<typename T>
        tensor(const std::vector<T>& values, const std::vector<int64_t>& shape);

        /**
        * Creates a flat tensor with the given values
        * @tparam T A type that can be convertible into a tensor
        * @param values The values to be converted
        */
        template<typename T>
        tensor(const std::initializer_list<T>& values);

        /**
         * Creates a tensor with the given value
         * @tparam T A type that can be convertible into a tensor
         * @param value The value to be converted
         */
        template<typename T>
        tensor(const T& value);

        /**
		* Creates a flat tensor with the given values
		* @tparam T A type that can be convertible into a tensor
		* @param values The values to be converted
		*/
		template<typename T>
        tensor(const std::vector<T>& values);

        /**
         * @return Shape of the tensor
         */
        tensor shape() const;

        /**
         * @param on_memory If false, the function will return the name of the device that produced the tensor.
         * If true, the function will return the name of the device in whose memory the tensor resides
         * @return Returns the name of the device of the tensor
         */
        std::string device(bool on_memory=false) const;


        /**
         * @return The tensor datatype
         */
        datatype dtype() const;

        /**
         * Converts the tensor into a C++ vector
         * @tparam T The c++ type (must be equivalent to the tensor type)
         * @return A vector representing the flat tensor
         */
        template<typename T>
        std::vector<T> get_data() const;


        ~tensor() = default;
        tensor(const tensor &tensor) = default;
        tensor(tensor &&tensor) = default;
        tensor &operator=(const tensor &other) = default;
        tensor &operator=(tensor &&other) = default;


        std::shared_ptr<TF_Tensor> tf_tensor;
        std::shared_ptr<TFE_TensorHandle> tfe_handle;

        explicit tensor(TFE_TensorHandle* handle);
        explicit tensor(TF_Tensor* t);

    private:

        tensor(enum TF_DataType type, const void* data, size_t len, const std::vector<int64_t>& shape);
    };
}


/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/


namespace cppflow {

    tensor::tensor(enum TF_DataType type, const void *data, size_t len, const std::vector<int64_t> &shape) {
        this->tf_tensor = {TF_AllocateTensor(type, shape.data(), (int)shape.size(), (int)len), TF_DeleteTensor};
        std::memcpy(TF_TensorData(this->tf_tensor.get()), data, TF_TensorByteSize(this->tf_tensor.get()));
        this->tfe_handle = {TFE_NewTensorHandle(this->tf_tensor.get(), context::get_status()), TFE_DeleteTensorHandle};
        status_check(context::get_status());
    }

    template<typename T>
    tensor::tensor(const std::vector<T>& values, const std::vector<int64_t>& shape) :
        tensor(deduce_tf_type<T>(), values.data(), values.size() * sizeof(T), shape) {}

    template<typename T>
    tensor::tensor(const std::initializer_list<T>& values) :
        tensor(std::vector<T>(values), {(int64_t)values.size()}) {}

    template<typename T>
    tensor::tensor(const T& value) :
        tensor(std::vector<T>({value}), {}) {}

	template<typename T>
	tensor::tensor(const std::vector<T>& values) :
		tensor(values, {(int64_t)values.size()}) {}

    // For future version TensorFlow 2.4
    //template<>
    //tensor::tensor(const std::string& value) {
    //    TF_TString tstr[1];
    //    TF_TString_Init(&tstr[0]);
    //    TF_TString_Copy(&tstr[0], value.c_str(), value.size());
    //
    //    *this = tensor(static_cast<enum TF_DataType>(TF_STRING), (void *) tstr, TF_TString_GetSize(tstr), {});
    //}

    template<>
    tensor::tensor(const std::string& value) {
        size_t size = 8 + TF_StringEncodedSize(value.length());
        std::vector<char> buffer(value.size() + 8);
        for (int i=0; i<8; i++) { buffer[i]=0;}
        TF_StringEncode(value.c_str(), value.size(), buffer.data() + 8, size - 8, context::get_status());
        status_check(context::get_status());

        *this = tensor(static_cast<enum TF_DataType>(TF_STRING), (void *)buffer.data(), size, {});
    }

    tensor::tensor(TFE_TensorHandle* handle) {
            this->tfe_handle = {handle, TFE_DeleteTensorHandle};
    }

    tensor::tensor(TF_Tensor* t) {
        this->tf_tensor = {t, TF_DeleteTensor};
        this->tfe_handle = {TFE_NewTensorHandle(this->tf_tensor.get(), context::get_status()), TFE_DeleteTensorHandle};
        status_check(context::get_status());
    }

    tensor tensor::shape() const {
        auto op = TFE_NewOp(context::get_context(), "Shape", context::get_status());
        status_check(context::get_status());

        TFE_OpAddInput(op, this->tfe_handle.get(), context::get_status());
        status_check(context::get_status());

        // EXECUTE
        int n = 1;
        TFE_TensorHandle* res[1];
        TFE_Execute(op, res, &n, context::get_status());
        status_check(context::get_status());
        TFE_DeleteOp(op);

        tensor r;
        r.tf_tensor = { TFE_TensorHandleResolve(res[0], context::get_status()), TF_DeleteTensor};
        status_check(context::get_status());
        r.tfe_handle = {TFE_NewTensorHandle(r.tf_tensor.get(), context::get_status()), TFE_DeleteTensorHandle};
        status_check(context::get_status());

        return r;
    }

    std::string tensor::device(bool on_memory) const {
        std::string res;
        if (on_memory)
            res = TFE_TensorHandleBackingDeviceName(this->tfe_handle.get(), context::get_status());
        else
            res = std::string(TFE_TensorHandleDeviceName(this->tfe_handle.get(), context::get_status()));

        status_check(context::get_status());
        return res;
    }

    template<typename T>
    std::vector<T> tensor::get_data() const {
        auto res_tensor = TFE_TensorHandleResolve(this->tfe_handle.get(), context::get_status());
        status_check(context::get_status());

        // Check tensor data is not empty
        auto raw_data = TF_TensorData(res_tensor);
        //this->error_check(raw_data != nullptr, "Tensor data is empty");

        size_t size = TF_TensorByteSize(res_tensor) / TF_DataTypeSize(TF_TensorType(res_tensor));

        // Convert to correct type
        const auto T_data = static_cast<T*>(raw_data);
        return std::vector<T>(T_data, T_data + size);
    }

    datatype tensor::dtype() const {
        return TFE_TensorHandleDataType(this->tfe_handle.get());
    }
}

#endif //CPPFLOW2_TENSOR_H
