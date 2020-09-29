
/**
 * @file ops.h
 * TensorFlow raw_ops mappings
 */

#ifndef CPPFLOW2_RAW_OPS_H
#define CPPFLOW2_RAW_OPS_H

#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>

#include <tensorflow/c/eager/c_api.h>
#include <tensorflow/c/tf_datatype.h>
#include <tensorflow/c/tf_tensor.h>

#include "tensor.h"
#include "datatype.h"

namespace cppflow {



tensor abs(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Abs", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor accumulate_n_v2(const std::vector<tensor>&inputs, const std::vector<int64_t>& shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AccumulateNV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> inputs_handles; inputs_handles.reserve((int)inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, inputs_handles.data(), (int)inputs.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", (int)inputs.size());
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor accumulator_num_accumulated(const tensor& handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AccumulatorNumAccumulated", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor accumulator_take_gradient(const tensor& handle, const tensor& num_required, datatype dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AccumulatorTakeGradient", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_required.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor acos(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Acos", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor acosh(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Acosh", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor add(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Add", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor add_many_sparse_to_tensors_map(const tensor& sparse_indices, const tensor& sparse_values, const tensor& sparse_shape, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AddManySparseToTensorsMap", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sparse_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sparse_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sparse_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor add_n(const std::vector<tensor>&inputs) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AddN", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> inputs_handles; inputs_handles.reserve((int)inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, inputs_handles.data(), (int)inputs.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", (int)inputs.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor add_sparse_to_tensors_map(const tensor& sparse_indices, const tensor& sparse_values, const tensor& sparse_shape, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AddSparseToTensorsMap", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sparse_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sparse_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sparse_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor add_v2(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AddV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor adjust_contrast(const tensor& images, const tensor& contrast_factor, const tensor& min_value, const tensor& max_value) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AdjustContrast", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, contrast_factor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, min_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor adjust_contrastv2(const tensor& images, const tensor& contrast_factor) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AdjustContrastv2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, contrast_factor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor adjust_hue(const tensor& images, const tensor& delta) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AdjustHue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, delta.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor adjust_saturation(const tensor& images, const tensor& scale) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AdjustSaturation", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, scale.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor all(const tensor& input, const tensor& reduction_indices, bool keep_dims=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "All", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor all_to_all(const tensor& input, const tensor& group_assignment, int64_t concat_dimension, int64_t split_dimension, int64_t split_count) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AllToAll", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, group_assignment.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "concat_dimension", concat_dimension);
    TFE_OpSetAttrInt(op, "split_dimension", split_dimension);
    TFE_OpSetAttrInt(op, "split_count", split_count);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor angle(const tensor& input, datatype Tout=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Angle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tout", Tout);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor anonymous_iterator(const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AnonymousIterator", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor any(const tensor& input, const tensor& reduction_indices, bool keep_dims=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Any", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_ada_max(const tensor& var, const tensor& m, const tensor& v, const tensor& beta1_power, const tensor& lr, const tensor& beta1, const tensor& beta2, const tensor& epsilon, const tensor& grad, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyAdaMax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, m.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, v.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta1_power.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_adadelta(const tensor& var, const tensor& accum, const tensor& accum_update, const tensor& lr, const tensor& rho, const tensor& epsilon, const tensor& grad, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyAdadelta", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum_update.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rho.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_adagrad(const tensor& var, const tensor& accum, const tensor& lr, const tensor& grad, bool use_locking=false, bool update_slots=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyAdagrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "update_slots", (unsigned char)update_slots);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_adagrad_d_a(const tensor& var, const tensor& gradient_accumulator, const tensor& gradient_squared_accumulator, const tensor& grad, const tensor& lr, const tensor& l1, const tensor& l2, const tensor& global_step, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyAdagradDA", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, gradient_accumulator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, gradient_squared_accumulator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, global_step.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_adagrad_v2(const tensor& var, const tensor& accum, const tensor& lr, const tensor& epsilon, const tensor& grad, bool use_locking=false, bool update_slots=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyAdagradV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "update_slots", (unsigned char)update_slots);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_adam(const tensor& var, const tensor& m, const tensor& v, const tensor& beta1_power, const tensor& beta2_power, const tensor& lr, const tensor& beta1, const tensor& beta2, const tensor& epsilon, const tensor& grad, bool use_locking=false, bool use_nesterov=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyAdam", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, m.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, v.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta1_power.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta2_power.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "use_nesterov", (unsigned char)use_nesterov);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_add_sign(const tensor& var, const tensor& m, const tensor& lr, const tensor& alpha, const tensor& sign_decay, const tensor& beta, const tensor& grad, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyAddSign", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, m.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, alpha.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sign_decay.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_centered_r_m_s_prop(const tensor& var, const tensor& mg, const tensor& ms, const tensor& mom, const tensor& lr, const tensor& rho, const tensor& momentum, const tensor& epsilon, const tensor& grad, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyCenteredRMSProp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, mg.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, ms.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, mom.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rho.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, momentum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_ftrl(const tensor& var, const tensor& accum, const tensor& linear, const tensor& grad, const tensor& lr, const tensor& l1, const tensor& l2, const tensor& lr_power, bool use_locking=false, bool multiply_linear_by_lr=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyFtrl", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, linear.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr_power.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "multiply_linear_by_lr", (unsigned char)multiply_linear_by_lr);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_ftrl_v2(const tensor& var, const tensor& accum, const tensor& linear, const tensor& grad, const tensor& lr, const tensor& l1, const tensor& l2, const tensor& l2_shrinkage, const tensor& lr_power, bool use_locking=false, bool multiply_linear_by_lr=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyFtrlV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, linear.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2_shrinkage.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr_power.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "multiply_linear_by_lr", (unsigned char)multiply_linear_by_lr);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_gradient_descent(const tensor& var, const tensor& alpha, const tensor& delta, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyGradientDescent", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, alpha.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, delta.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_momentum(const tensor& var, const tensor& accum, const tensor& lr, const tensor& grad, const tensor& momentum, bool use_locking=false, bool use_nesterov=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyMomentum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, momentum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "use_nesterov", (unsigned char)use_nesterov);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_power_sign(const tensor& var, const tensor& m, const tensor& lr, const tensor& logbase, const tensor& sign_decay, const tensor& beta, const tensor& grad, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyPowerSign", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, m.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, logbase.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sign_decay.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_proximal_adagrad(const tensor& var, const tensor& accum, const tensor& lr, const tensor& l1, const tensor& l2, const tensor& grad, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyProximalAdagrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_proximal_gradient_descent(const tensor& var, const tensor& alpha, const tensor& l1, const tensor& l2, const tensor& delta, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyProximalGradientDescent", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, alpha.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, delta.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor apply_r_m_s_prop(const tensor& var, const tensor& ms, const tensor& mom, const tensor& lr, const tensor& rho, const tensor& momentum, const tensor& epsilon, const tensor& grad, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApplyRMSProp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, ms.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, mom.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rho.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, momentum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor approximate_equal(const tensor& x, const tensor& y, float tolerance=1.0000e-05) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ApproximateEqual", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrFloat(op, "tolerance", tolerance);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor arg_max(const tensor& input, const tensor& dimension, datatype Tidx=static_cast<datatype>(3), datatype output_type=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ArgMax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dimension.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);
    TFE_OpSetAttrType(op, "output_type", output_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor arg_min(const tensor& input, const tensor& dimension, datatype Tidx=static_cast<datatype>(3), datatype output_type=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ArgMin", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dimension.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);
    TFE_OpSetAttrType(op, "output_type", output_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor as_string(const tensor& input, int64_t precision=-1, bool scientific=false, bool shortest=false, int64_t width=-1, const std::string& fill="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AsString", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "precision", precision);
    TFE_OpSetAttrBool(op, "scientific", (unsigned char)scientific);
    TFE_OpSetAttrBool(op, "shortest", (unsigned char)shortest);
    TFE_OpSetAttrInt(op, "width", width);
    TFE_OpSetAttrString(op, "fill", (void*) fill.c_str(), fill.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor asin(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Asin", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor asinh(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Asinh", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor assert_cardinality_dataset(const tensor& input_dataset, const tensor& cardinality, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AssertCardinalityDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, cardinality.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor assert_next_dataset(const tensor& input_dataset, const tensor& transformations, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AssertNextDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, transformations.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor assign(const tensor& ref, const tensor& value, bool validate_shape=true, bool use_locking=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Assign", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "validate_shape", (unsigned char)validate_shape);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor assign_add(const tensor& ref, const tensor& value, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AssignAdd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor assign_sub(const tensor& ref, const tensor& value, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AssignSub", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor atan(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Atan", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor atan2(const tensor& y, const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Atan2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor atanh(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Atanh", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor audio_spectrogram(const tensor& input, int64_t window_size, int64_t stride, bool magnitude_squared=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AudioSpectrogram", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "window_size", window_size);
    TFE_OpSetAttrInt(op, "stride", stride);
    TFE_OpSetAttrBool(op, "magnitude_squared", (unsigned char)magnitude_squared);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor audio_summary(const tensor& tag, const tensor& input_tensor, float sample_rate, int64_t max_outputs=3) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AudioSummary", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrFloat(op, "sample_rate", sample_rate);
    TFE_OpSetAttrInt(op, "max_outputs", max_outputs);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor audio_summary_v2(const tensor& tag, const tensor& input_tensor, const tensor& sample_rate, int64_t max_outputs=3) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AudioSummaryV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sample_rate.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "max_outputs", max_outputs);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor auto_shard_dataset(const tensor& input_dataset, const tensor& num_workers, const tensor& index, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, int64_t auto_shard_policy=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AutoShardDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_workers.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "auto_shard_policy", auto_shard_policy);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor avg_pool(const tensor& value, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AvgPool", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor avg_pool3_d(const tensor& input, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, const std::string& data_format="NDHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AvgPool3D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor avg_pool3_d_grad(const tensor& orig_input_shape, const tensor& grad, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, const std::string& data_format="NDHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AvgPool3DGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, orig_input_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor avg_pool_grad(const tensor& orig_input_shape, const tensor& grad, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "AvgPoolGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, orig_input_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor barrier(const std::vector<datatype>& component_types, const std::vector< std::vector<int64_t>>& shapes, int64_t capacity=-1, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Barrier", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor barrier_incomplete_size(const tensor& handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BarrierIncompleteSize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor barrier_ready_size(const tensor& handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BarrierReadySize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_cholesky(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchCholesky", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_cholesky_grad(const tensor& l, const tensor& grad) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchCholeskyGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, l.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_dataset(const tensor& input_dataset, const tensor& batch_size, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, batch_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_dataset_v2(const tensor& input_dataset, const tensor& batch_size, const tensor& drop_remainder, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, bool parallel_copy=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchDatasetV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, batch_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, drop_remainder.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrBool(op, "parallel_copy", (unsigned char)parallel_copy);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_f_f_t(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchFFT", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_f_f_t2_d(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchFFT2D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_f_f_t3_d(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchFFT3D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_i_f_f_t(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchIFFT", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_i_f_f_t2_d(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchIFFT2D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_i_f_f_t3_d(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchIFFT3D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_mat_mul(const tensor& x, const tensor& y, bool adj_x=false, bool adj_y=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatMul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "adj_x", (unsigned char)adj_x);
    TFE_OpSetAttrBool(op, "adj_y", (unsigned char)adj_y);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_mat_mul_v2(const tensor& x, const tensor& y, bool adj_x=false, bool adj_y=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatMulV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "adj_x", (unsigned char)adj_x);
    TFE_OpSetAttrBool(op, "adj_y", (unsigned char)adj_y);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_matrix_band_part(const tensor& input, const tensor& num_lower, const tensor& num_upper) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatrixBandPart", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_lower.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_upper.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_matrix_determinant(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatrixDeterminant", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_matrix_diag(const tensor& diagonal) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatrixDiag", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, diagonal.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_matrix_diag_part(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatrixDiagPart", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_matrix_inverse(const tensor& input, bool adjoint=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatrixInverse", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "adjoint", (unsigned char)adjoint);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_matrix_set_diag(const tensor& input, const tensor& diagonal) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatrixSetDiag", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, diagonal.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_matrix_solve(const tensor& matrix, const tensor& rhs, bool adjoint=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatrixSolve", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, matrix.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rhs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "adjoint", (unsigned char)adjoint);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_matrix_solve_ls(const tensor& matrix, const tensor& rhs, const tensor& l2_regularizer, bool fast=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatrixSolveLs", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, matrix.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rhs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2_regularizer.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "fast", (unsigned char)fast);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_matrix_triangular_solve(const tensor& matrix, const tensor& rhs, bool lower=true, bool adjoint=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchMatrixTriangularSolve", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, matrix.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rhs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "lower", (unsigned char)lower);
    TFE_OpSetAttrBool(op, "adjoint", (unsigned char)adjoint);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_norm_with_global_normalization(const tensor& t, const tensor& m, const tensor& v, const tensor& beta, const tensor& gamma, float variance_epsilon, bool scale_after_normalization) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchNormWithGlobalNormalization", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, t.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, m.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, v.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, gamma.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrFloat(op, "variance_epsilon", variance_epsilon);
    TFE_OpSetAttrBool(op, "scale_after_normalization", (unsigned char)scale_after_normalization);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_self_adjoint_eig(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchSelfAdjointEig", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_to_space(const tensor& input, const tensor& crops, int64_t block_size, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchToSpace", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, crops.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "block_size", block_size);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor batch_to_space_n_d(const tensor& input, const tensor& block_shape, const tensor& crops, datatype Tblock_shape=static_cast<datatype>(3), datatype Tcrops=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BatchToSpaceND", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, block_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, crops.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tblock_shape", Tblock_shape);
    TFE_OpSetAttrType(op, "Tcrops", Tcrops);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bessel_i0e(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BesselI0e", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bessel_i1e(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BesselI1e", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor betainc(const tensor& a, const tensor& b, const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Betainc", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bias_add(const tensor& value, const tensor& bias, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BiasAdd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, bias.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bias_add_grad(const tensor& out_backprop, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BiasAddGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bias_add_v1(const tensor& value, const tensor& bias) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BiasAddV1", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, bias.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bincount(const tensor& arr, const tensor& size, const tensor& weights) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Bincount", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, arr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, weights.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bitcast(const tensor& input, datatype type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Bitcast", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "type", type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bitwise_and(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BitwiseAnd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bitwise_or(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BitwiseOr", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bitwise_xor(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BitwiseXor", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_aggregate_stats(const tensor& node_ids, const tensor& gradients, const tensor& hessians, const tensor& feature, int64_t max_splits, int64_t num_buckets) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesAggregateStats", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, node_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, hessians.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, feature.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "max_splits", max_splits);
    TFE_OpSetAttrInt(op, "num_buckets", num_buckets);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_bucketize(const std::vector<tensor>&float_values, const std::vector<tensor>&bucket_boundaries) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesBucketize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> float_values_handles; float_values_handles.reserve((int)float_values.size());
    std::transform(float_values.begin(), float_values.end(), std::back_inserter(float_values_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, float_values_handles.data(), (int)float_values.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> bucket_boundaries_handles; bucket_boundaries_handles.reserve((int)bucket_boundaries.size());
    std::transform(bucket_boundaries.begin(), bucket_boundaries.end(), std::back_inserter(bucket_boundaries_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, bucket_boundaries_handles.data(), (int)bucket_boundaries.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_features", (int)float_values.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_center_bias(const tensor& tree_ensemble_handle, const tensor& mean_gradients, const tensor& mean_hessians, const tensor& l1, const tensor& l2) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesCenterBias", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, tree_ensemble_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, mean_gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, mean_hessians.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_ensemble_resource_handle_op(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesEnsembleResourceHandleOp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_example_debug_outputs(const tensor& tree_ensemble_handle, const std::vector<tensor>&bucketized_features, int64_t logits_dimension) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesExampleDebugOutputs", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, tree_ensemble_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> bucketized_features_handles; bucketized_features_handles.reserve((int)bucketized_features.size());
    std::transform(bucketized_features.begin(), bucketized_features.end(), std::back_inserter(bucketized_features_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, bucketized_features_handles.data(), (int)bucketized_features.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_bucketized_features", (int)bucketized_features.size());
    TFE_OpSetAttrInt(op, "logits_dimension", logits_dimension);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_flush_quantile_summaries(const tensor& quantile_stream_resource_handle, int64_t num_features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesFlushQuantileSummaries", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, quantile_stream_resource_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_features", num_features);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_make_quantile_summaries(const std::vector<tensor>&float_values, const tensor& example_weights, const tensor& epsilon) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesMakeQuantileSummaries", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> float_values_handles; float_values_handles.reserve((int)float_values.size());
    std::transform(float_values.begin(), float_values.end(), std::back_inserter(float_values_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, float_values_handles.data(), (int)float_values.size(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, example_weights.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_features", (int)float_values.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_make_stats_summary(const tensor& node_ids, const tensor& gradients, const tensor& hessians, const std::vector<tensor>&bucketized_features_list, int64_t max_splits, int64_t num_buckets) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesMakeStatsSummary", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, node_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, hessians.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> bucketized_features_list_handles; bucketized_features_list_handles.reserve((int)bucketized_features_list.size());
    std::transform(bucketized_features_list.begin(), bucketized_features_list.end(), std::back_inserter(bucketized_features_list_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, bucketized_features_list_handles.data(), (int)bucketized_features_list.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "max_splits", max_splits);
    TFE_OpSetAttrInt(op, "num_buckets", num_buckets);
    TFE_OpSetAttrInt(op, "num_features", (int)bucketized_features_list.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_predict(const tensor& tree_ensemble_handle, const std::vector<tensor>&bucketized_features, int64_t logits_dimension) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesPredict", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, tree_ensemble_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> bucketized_features_handles; bucketized_features_handles.reserve((int)bucketized_features.size());
    std::transform(bucketized_features.begin(), bucketized_features.end(), std::back_inserter(bucketized_features_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, bucketized_features_handles.data(), (int)bucketized_features.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_bucketized_features", (int)bucketized_features.size());
    TFE_OpSetAttrInt(op, "logits_dimension", logits_dimension);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_quantile_stream_resource_get_bucket_boundaries(const tensor& quantile_stream_resource_handle, int64_t num_features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesQuantileStreamResourceGetBucketBoundaries", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, quantile_stream_resource_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_features", num_features);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor boosted_trees_quantile_stream_resource_handle_op(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BoostedTreesQuantileStreamResourceHandleOp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor broadcast_args(const tensor& s0, const tensor& s1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BroadcastArgs", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, s0.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, s1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor broadcast_to(const tensor& input, const tensor& shape, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BroadcastTo", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bucketize(const tensor& input, const std::vector<float>& boundaries) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Bucketize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrFloatList(op, "boundaries", boundaries.data(), (int)boundaries.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor bytes_produced_stats_dataset(const tensor& input_dataset, const tensor& tag, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "BytesProducedStatsDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor c_s_r_sparse_matrix_to_dense(const tensor& sparse_input, datatype type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CSRSparseMatrixToDense", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sparse_input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "type", type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor c_s_v_dataset(const tensor& filenames, const tensor& compression_type, const tensor& buffer_size, const tensor& header, const tensor& field_delim, const tensor& use_quote_delim, const tensor& na_value, const tensor& select_cols, const std::vector<tensor>&record_defaults, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CSVDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, filenames.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, compression_type.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, header.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, field_delim.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, use_quote_delim.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, na_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, select_cols.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> record_defaults_handles; record_defaults_handles.reserve((int)record_defaults.size());
    std::transform(record_defaults.begin(), record_defaults.end(), std::back_inserter(record_defaults_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, record_defaults_handles.data(), (int)record_defaults.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cache_dataset(const tensor& input_dataset, const tensor& filename, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CacheDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filename.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cache_dataset_v2(const tensor& input_dataset, const tensor& filename, const tensor& cache, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CacheDatasetV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filename.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, cache.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cast(const tensor& x, datatype SrcT, datatype DstT, bool Truncate=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Cast", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "SrcT", SrcT);
    TFE_OpSetAttrType(op, "DstT", DstT);
    TFE_OpSetAttrBool(op, "Truncate", (unsigned char)Truncate);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ceil(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Ceil", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor check_numerics(const tensor& input_tensor, const std::string& message) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CheckNumerics", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "message", (void*) message.c_str(), message.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor check_numerics_v2(const tensor& input_tensor, const std::string& message) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CheckNumericsV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "message", (void*) message.c_str(), message.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cholesky(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Cholesky", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cholesky_grad(const tensor& l, const tensor& grad) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CholeskyGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, l.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor choose_fastest_dataset(const std::vector<tensor>&input_datasets, int64_t num_experiments, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ChooseFastestDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> input_datasets_handles; input_datasets_handles.reserve((int)input_datasets.size());
    std::transform(input_datasets.begin(), input_datasets.end(), std::back_inserter(input_datasets_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, input_datasets_handles.data(), (int)input_datasets.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", (int)input_datasets.size());
    TFE_OpSetAttrInt(op, "num_experiments", num_experiments);
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor clip_by_value(const tensor& t, const tensor& clip_value_min, const tensor& clip_value_max) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ClipByValue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, t.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, clip_value_min.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, clip_value_max.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor collective_bcast_recv(int64_t group_size, int64_t group_key, int64_t instance_key, const std::vector<int64_t>& shape, const std::string& communication_hint="auto", float timeout_seconds=0.0000e+00) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CollectiveBcastRecv", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrInt(op, "group_size", group_size);
    TFE_OpSetAttrInt(op, "group_key", group_key);
    TFE_OpSetAttrInt(op, "instance_key", instance_key);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "communication_hint", (void*) communication_hint.c_str(), communication_hint.size());
    TFE_OpSetAttrFloat(op, "timeout_seconds", timeout_seconds);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor collective_bcast_send(const tensor& input, int64_t group_size, int64_t group_key, int64_t instance_key, const std::vector<int64_t>& shape, const std::string& communication_hint="auto", float timeout_seconds=0.0000e+00) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CollectiveBcastSend", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "group_size", group_size);
    TFE_OpSetAttrInt(op, "group_key", group_key);
    TFE_OpSetAttrInt(op, "instance_key", instance_key);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "communication_hint", (void*) communication_hint.c_str(), communication_hint.size());
    TFE_OpSetAttrFloat(op, "timeout_seconds", timeout_seconds);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor collective_gather(const tensor& input, int64_t group_size, int64_t group_key, int64_t instance_key, const std::vector<int64_t>& shape, const std::string& communication_hint="auto", float timeout_seconds=0.0000e+00) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CollectiveGather", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "group_size", group_size);
    TFE_OpSetAttrInt(op, "group_key", group_key);
    TFE_OpSetAttrInt(op, "instance_key", instance_key);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "communication_hint", (void*) communication_hint.c_str(), communication_hint.size());
    TFE_OpSetAttrFloat(op, "timeout_seconds", timeout_seconds);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor collective_permute(const tensor& input, const tensor& source_target_pairs) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CollectivePermute", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, source_target_pairs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor collective_reduce(const tensor& input, int64_t group_size, int64_t group_key, int64_t instance_key, const std::string& merge_op, const std::string& final_op, const std::vector<int64_t>& subdiv_offsets, const std::vector<int64_t>& wait_for, const std::string& communication_hint="auto", float timeout_seconds=0.0000e+00) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CollectiveReduce", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "group_size", group_size);
    TFE_OpSetAttrInt(op, "group_key", group_key);
    TFE_OpSetAttrInt(op, "instance_key", instance_key);
    TFE_OpSetAttrString(op, "merge_op", (void*) merge_op.c_str(), (int)merge_op.size());
    TFE_OpSetAttrString(op, "final_op", (void*) final_op.c_str(), (int)final_op.size());
    TFE_OpSetAttrIntList(op, "subdiv_offsets", subdiv_offsets.data(), (int)subdiv_offsets.size());
    TFE_OpSetAttrIntList(op, "wait_for", wait_for.data(), (int)wait_for.size());
    TFE_OpSetAttrString(op, "communication_hint", (void*) communication_hint.c_str(), (int)communication_hint.size());
    TFE_OpSetAttrFloat(op, "timeout_seconds", timeout_seconds);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor compare_and_bitpack(const tensor& input, const tensor& threshold) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CompareAndBitpack", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, threshold.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor complex(const tensor& real, const tensor& imag, datatype Tout=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Complex", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, real.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, imag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tout", Tout);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor complex_abs(const tensor& x, datatype Tout=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ComplexAbs", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tout", Tout);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor concat(const tensor& concat_dim, const std::vector<tensor>&values) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Concat", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, concat_dim.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> values_handles; values_handles.reserve((int)values.size());
    std::transform(values.begin(), values.end(), std::back_inserter(values_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, values_handles.data(), (int)values.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", values.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor concat_offset(const tensor& concat_dim, const std::vector<tensor>&shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ConcatOffset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, concat_dim.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> shape_handles; shape_handles.reserve((int)shape.size());
    std::transform(shape.begin(), shape.end(), std::back_inserter(shape_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, shape_handles.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", (int)shape.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor concat_v2(const std::vector<tensor>&values, const tensor& axis, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ConcatV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> values_handles; values_handles.reserve((int)values.size());
    std::transform(values.begin(), values.end(), std::back_inserter(values_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, values_handles.data(), (int)values.size(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, axis.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", values.size());
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor concatenate_dataset(const tensor& input_dataset, const tensor& another_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ConcatenateDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, another_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conditional_accumulator(datatype dtype, const std::vector<int64_t>& shape, const std::string& container="", const std::string& shared_name="", const std::string& reduction_type="MEAN") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ConditionalAccumulator", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrString(op, "reduction_type", (void*) reduction_type.c_str(), reduction_type.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor configure_distributed_t_p_u(const std::string& embedding_config="", const std::string& tpu_embedding_config="", bool is_global_init=false, bool enable_whole_mesh_compilations=false, bool compilation_failure_closes_chips=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ConfigureDistributedTPU", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "embedding_config", (void*) embedding_config.c_str(), embedding_config.size());
    TFE_OpSetAttrString(op, "tpu_embedding_config", (void*) tpu_embedding_config.c_str(), tpu_embedding_config.size());
    TFE_OpSetAttrBool(op, "is_global_init", (unsigned char)is_global_init);
    TFE_OpSetAttrBool(op, "enable_whole_mesh_compilations", (unsigned char)enable_whole_mesh_compilations);
    TFE_OpSetAttrBool(op, "compilation_failure_closes_chips", (unsigned char)compilation_failure_closes_chips);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conj(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Conj", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conjugate_transpose(const tensor& x, const tensor& perm, datatype Tperm=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ConjugateTranspose", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, perm.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tperm", Tperm);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor const_tensor(const tensor& value, datatype dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Const", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    
    TFE_OpSetAttrTensor(op, "value", value.tf_tensor.get(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conv2_d(const tensor& input, const tensor& filter, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& explicit_paddings, const std::vector<int64_t>& dilations, bool use_cudnn_on_gpu=true, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Conv2D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());
    TFE_OpSetAttrIntList(op, "explicit_paddings", explicit_paddings.data(), (int)explicit_paddings.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());
    TFE_OpSetAttrBool(op, "use_cudnn_on_gpu", (unsigned char)use_cudnn_on_gpu);
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conv2_d_backprop_filter(const tensor& input, const tensor& filter_sizes, const tensor& out_backprop, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& explicit_paddings, const std::vector<int64_t>& dilations, bool use_cudnn_on_gpu=true, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Conv2DBackpropFilter", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter_sizes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrIntList(op, "explicit_paddings", explicit_paddings.data(), (int)explicit_paddings.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());
    TFE_OpSetAttrBool(op, "use_cudnn_on_gpu", (unsigned char)use_cudnn_on_gpu);
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conv2_d_backprop_input(const tensor& input_sizes, const tensor& filter, const tensor& out_backprop, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& explicit_paddings, const std::vector<int64_t>& dilations, bool use_cudnn_on_gpu=true, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Conv2DBackpropInput", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_sizes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrIntList(op, "explicit_paddings", explicit_paddings.data(), (int)explicit_paddings.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());
    TFE_OpSetAttrBool(op, "use_cudnn_on_gpu", (unsigned char)use_cudnn_on_gpu);
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conv3_d(const tensor& input, const tensor& filter, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& dilations, const std::string& data_format="NDHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Conv3D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conv3_d_backprop_filter(const tensor& input, const tensor& filter, const tensor& out_backprop, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& dilations) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Conv3DBackpropFilter", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conv3_d_backprop_filter_v2(const tensor& input, const tensor& filter_sizes, const tensor& out_backprop, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& dilations, const std::string& data_format="NDHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Conv3DBackpropFilterV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter_sizes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conv3_d_backprop_input(const tensor& input, const tensor& filter, const tensor& out_backprop, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& dilations) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Conv3DBackpropInput", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor conv3_d_backprop_input_v2(const tensor& input_sizes, const tensor& filter, const tensor& out_backprop, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& dilations, const std::string& data_format="NDHWC", datatype Tshape=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Conv3DBackpropInputV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_sizes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());
    TFE_OpSetAttrType(op, "Tshape", Tshape);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor copy(const tensor& input, const std::vector< std::string>& debug_ops_spec, const std::string& tensor_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Copy", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> debug_ops_spec_sizes; debug_ops_spec_sizes.reserve(debug_ops_spec.size());
    std::transform(debug_ops_spec.begin(), debug_ops_spec.end(), std::back_inserter(debug_ops_spec_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "debug_ops_spec", reinterpret_cast<const void *const *>(debug_ops_spec.data()), debug_ops_spec_sizes.data(), (int)debug_ops_spec.size());
    
    TFE_OpSetAttrString(op, "tensor_name", (void*) tensor_name.c_str(), tensor_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor copy_host(const tensor& input, const std::vector< std::string>& debug_ops_spec, const std::string& tensor_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CopyHost", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> debug_ops_spec_sizes; debug_ops_spec_sizes.reserve(debug_ops_spec.size());
    std::transform(debug_ops_spec.begin(), debug_ops_spec.end(), std::back_inserter(debug_ops_spec_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "debug_ops_spec", reinterpret_cast<const void *const *>(debug_ops_spec.data()), debug_ops_spec_sizes.data(), (int)debug_ops_spec.size());
    
    TFE_OpSetAttrString(op, "tensor_name", (void*) tensor_name.c_str(), tensor_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cos(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Cos", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cosh(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Cosh", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor count_up_to(const tensor& ref, int64_t limit) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CountUpTo", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "limit", limit);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor crop_and_resize(const tensor& image, const tensor& boxes, const tensor& box_ind, const tensor& crop_size, const std::string& method="bilinear", float extrapolation_value=0.0000e+00) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CropAndResize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, image.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, boxes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, box_ind.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, crop_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "method", (void*) method.c_str(), method.size());
    TFE_OpSetAttrFloat(op, "extrapolation_value", extrapolation_value);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor crop_and_resize_grad_boxes(const tensor& grads, const tensor& image, const tensor& boxes, const tensor& box_ind, const std::string& method="bilinear") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CropAndResizeGradBoxes", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, grads.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, image.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, boxes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, box_ind.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "method", (void*) method.c_str(), method.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor crop_and_resize_grad_image(const tensor& grads, const tensor& boxes, const tensor& box_ind, const tensor& image_size, const std::string& method="bilinear") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CropAndResizeGradImage", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, grads.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, boxes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, box_ind.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, image_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "method", (void*) method.c_str(), method.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cross(const tensor& a, const tensor& b) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Cross", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cross_replica_sum(const tensor& input, const tensor& group_assignment) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CrossReplicaSum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, group_assignment.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cudnn_r_n_n_canonical_to_params(const tensor& num_layers, const tensor& num_units, const tensor& input_size, const std::vector<tensor>&weights, const std::vector<tensor>&biases, const std::string& rnn_mode="lstm", const std::string& input_mode="linear_input", const std::string& direction="unidirectional", float dropout=0.0000e+00, int64_t seed=0, int64_t seed2=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CudnnRNNCanonicalToParams", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, num_layers.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_units.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> weights_handles; weights_handles.reserve((int)weights.size());
    std::transform(weights.begin(), weights.end(), std::back_inserter(weights_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, weights_handles.data(), (int)weights.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> biases_handles; biases_handles.reserve((int)biases.size());
    std::transform(biases.begin(), biases.end(), std::back_inserter(biases_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, biases_handles.data(), (int)biases.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_params", weights.size());
    TFE_OpSetAttrString(op, "rnn_mode", (void*) rnn_mode.c_str(), rnn_mode.size());
    TFE_OpSetAttrString(op, "input_mode", (void*) input_mode.c_str(), input_mode.size());
    TFE_OpSetAttrString(op, "direction", (void*) direction.c_str(), direction.size());
    TFE_OpSetAttrFloat(op, "dropout", dropout);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cudnn_r_n_n_canonical_to_params_v2(const tensor& num_layers, const tensor& num_units, const tensor& input_size, const std::vector<tensor>&weights, const std::vector<tensor>&biases, const std::string& rnn_mode="lstm", const std::string& input_mode="linear_input", const std::string& direction="unidirectional", float dropout=0.0000e+00, int64_t seed=0, int64_t seed2=0, int64_t num_proj=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CudnnRNNCanonicalToParamsV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, num_layers.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_units.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> weights_handles; weights_handles.reserve(weights.size());
    std::transform(weights.begin(), weights.end(), std::back_inserter(weights_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, weights_handles.data(), (int)weights.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> biases_handles; biases_handles.reserve((int)biases.size());
    std::transform(biases.begin(), biases.end(), std::back_inserter(biases_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, biases_handles.data(), (int)biases.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_params_weights", (int)weights.size());
    TFE_OpSetAttrInt(op, "num_params_biases", (int)biases.size());
    TFE_OpSetAttrString(op, "rnn_mode", (void*) rnn_mode.c_str(), (int)rnn_mode.size());
    TFE_OpSetAttrString(op, "input_mode", (void*) input_mode.c_str(), (int)input_mode.size());
    TFE_OpSetAttrString(op, "direction", (void*) direction.c_str(), (int)direction.size());
    TFE_OpSetAttrFloat(op, "dropout", dropout);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);
    TFE_OpSetAttrInt(op, "num_proj", num_proj);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cudnn_r_n_n_params_size(const tensor& num_layers, const tensor& num_units, const tensor& input_size, datatype S, const std::string& rnn_mode="lstm", const std::string& input_mode="linear_input", const std::string& direction="unidirectional", float dropout=0.0000e+00, int64_t seed=0, int64_t seed2=0, int64_t num_proj=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CudnnRNNParamsSize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, num_layers.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_units.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "S", S);
    TFE_OpSetAttrString(op, "rnn_mode", (void*) rnn_mode.c_str(), rnn_mode.size());
    TFE_OpSetAttrString(op, "input_mode", (void*) input_mode.c_str(), input_mode.size());
    TFE_OpSetAttrString(op, "direction", (void*) direction.c_str(), direction.size());
    TFE_OpSetAttrFloat(op, "dropout", dropout);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);
    TFE_OpSetAttrInt(op, "num_proj", num_proj);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cumprod(const tensor& x, const tensor& axis, bool exclusive=false, bool reverse=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Cumprod", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, axis.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "exclusive", (unsigned char)exclusive);
    TFE_OpSetAttrBool(op, "reverse", (unsigned char)reverse);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cumsum(const tensor& x, const tensor& axis, bool exclusive=false, bool reverse=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Cumsum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, axis.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "exclusive", (unsigned char)exclusive);
    TFE_OpSetAttrBool(op, "reverse", (unsigned char)reverse);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor cumulative_logsumexp(const tensor& x, const tensor& axis, bool exclusive=false, bool reverse=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "CumulativeLogsumexp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, axis.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "exclusive", (unsigned char)exclusive);
    TFE_OpSetAttrBool(op, "reverse", (unsigned char)reverse);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor data_format_dim_map(const tensor& x, const std::string& src_format="NHWC", const std::string& dst_format="NCHW") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DataFormatDimMap", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "src_format", (void*) src_format.c_str(), src_format.size());
    TFE_OpSetAttrString(op, "dst_format", (void*) dst_format.c_str(), dst_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor data_format_vec_permute(const tensor& x, const std::string& src_format="NHWC", const std::string& dst_format="NCHW") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DataFormatVecPermute", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "src_format", (void*) src_format.c_str(), src_format.size());
    TFE_OpSetAttrString(op, "dst_format", (void*) dst_format.c_str(), dst_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dataset_cardinality(const tensor& input_dataset) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DatasetCardinality", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dataset_from_graph(const tensor& graph_def) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DatasetFromGraph", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, graph_def.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dataset_to_graph(const tensor& input_dataset, const std::vector< std::string>& stateful_whitelist, bool allow_stateful=false, bool strip_device_assignment=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DatasetToGraph", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> stateful_whitelist_sizes; stateful_whitelist_sizes.reserve(stateful_whitelist.size());
    std::transform(stateful_whitelist.begin(), stateful_whitelist.end(), std::back_inserter(stateful_whitelist_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "stateful_whitelist", reinterpret_cast<const void *const *>(stateful_whitelist.data()), stateful_whitelist_sizes.data(), (int)stateful_whitelist.size());
    
    TFE_OpSetAttrBool(op, "allow_stateful", (unsigned char)allow_stateful);
    TFE_OpSetAttrBool(op, "strip_device_assignment", (unsigned char)strip_device_assignment);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dataset_to_graph_v2(const tensor& input_dataset, int64_t external_state_policy=0, bool strip_device_assignment=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DatasetToGraphV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "external_state_policy", external_state_policy);
    TFE_OpSetAttrBool(op, "strip_device_assignment", (unsigned char)strip_device_assignment);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dataset_to_single_element(const tensor& dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DatasetToSingleElement", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dawsn(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Dawsn", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor debug_gradient_identity(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DebugGradientIdentity", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor debug_gradient_ref_identity(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DebugGradientRefIdentity", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor debug_identity(const tensor& input, const std::vector< std::string>& debug_urls, const std::string& device_name="", const std::string& tensor_name="", bool gated_grpc=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DebugIdentity", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> debug_urls_sizes; debug_urls_sizes.reserve((int)debug_urls.size());
    std::transform(debug_urls.begin(), debug_urls.end(), std::back_inserter(debug_urls_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "debug_urls", reinterpret_cast<const void *const *>(debug_urls.data()), debug_urls_sizes.data(), (int)debug_urls.size());
    
    TFE_OpSetAttrString(op, "device_name", (void*) device_name.c_str(), device_name.size());
    TFE_OpSetAttrString(op, "tensor_name", (void*) tensor_name.c_str(), tensor_name.size());
    TFE_OpSetAttrBool(op, "gated_grpc", (unsigned char)gated_grpc);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor debug_identity_v2(const tensor& input, const std::vector< std::string>& debug_urls, const std::string& tfdbg_context_id="", const std::string& op_name="", int64_t output_slot=-1, int64_t tensor_debug_mode=-1, int64_t circular_buffer_size=1000, const std::string& tfdbg_run_id="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DebugIdentityV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> debug_urls_sizes; debug_urls_sizes.reserve(debug_urls.size());
    std::transform(debug_urls.begin(), debug_urls.end(), std::back_inserter(debug_urls_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "debug_urls", reinterpret_cast<const void *const *>(debug_urls.data()), debug_urls_sizes.data(), (int)debug_urls.size());
    
    TFE_OpSetAttrString(op, "tfdbg_context_id", (void*) tfdbg_context_id.c_str(), tfdbg_context_id.size());
    TFE_OpSetAttrString(op, "op_name", (void*) op_name.c_str(), op_name.size());
    TFE_OpSetAttrInt(op, "output_slot", output_slot);
    TFE_OpSetAttrInt(op, "tensor_debug_mode", tensor_debug_mode);
    TFE_OpSetAttrInt(op, "circular_buffer_size", circular_buffer_size);
    TFE_OpSetAttrString(op, "tfdbg_run_id", (void*) tfdbg_run_id.c_str(), tfdbg_run_id.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor debug_nan_count(const tensor& input, const std::vector< std::string>& debug_urls, const std::string& device_name="", const std::string& tensor_name="", bool gated_grpc=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DebugNanCount", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> debug_urls_sizes; debug_urls_sizes.reserve(debug_urls.size());
    std::transform(debug_urls.begin(), debug_urls.end(), std::back_inserter(debug_urls_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "debug_urls", reinterpret_cast<const void *const *>(debug_urls.data()), debug_urls_sizes.data(), (int)debug_urls.size());
    
    TFE_OpSetAttrString(op, "device_name", (void*) device_name.c_str(), device_name.size());
    TFE_OpSetAttrString(op, "tensor_name", (void*) tensor_name.c_str(), tensor_name.size());
    TFE_OpSetAttrBool(op, "gated_grpc", (unsigned char)gated_grpc);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor debug_numeric_summary(const tensor& input, const std::vector< std::string>& debug_urls, const std::string& device_name="", const std::string& tensor_name="", float lower_bound=-std::numeric_limits<float>::infinity(), float upper_bound=std::numeric_limits<float>::infinity(), bool mute_if_healthy=false, bool gated_grpc=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DebugNumericSummary", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> debug_urls_sizes; debug_urls_sizes.reserve((int)debug_urls.size());
    std::transform(debug_urls.begin(), debug_urls.end(), std::back_inserter(debug_urls_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "debug_urls", reinterpret_cast<const void *const *>(debug_urls.data()), debug_urls_sizes.data(), (int)debug_urls.size());
    
    TFE_OpSetAttrString(op, "device_name", (void*) device_name.c_str(), (int)device_name.size());
    TFE_OpSetAttrString(op, "tensor_name", (void*) tensor_name.c_str(), (int)tensor_name.size());
    TFE_OpSetAttrFloat(op, "lower_bound", lower_bound);
    TFE_OpSetAttrFloat(op, "upper_bound", upper_bound);
    TFE_OpSetAttrBool(op, "mute_if_healthy", (unsigned char)mute_if_healthy);
    TFE_OpSetAttrBool(op, "gated_grpc", (unsigned char)gated_grpc);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor debug_numeric_summary_v2(const tensor& input, datatype output_dtype=static_cast<datatype>(1), int64_t tensor_debug_mode=-1, int64_t tensor_id=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DebugNumericSummaryV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "output_dtype", output_dtype);
    TFE_OpSetAttrInt(op, "tensor_debug_mode", tensor_debug_mode);
    TFE_OpSetAttrInt(op, "tensor_id", tensor_id);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_and_crop_jpeg(const tensor& contents, const tensor& crop_window, int64_t channels=0, int64_t ratio=1, bool fancy_upscaling=true, bool try_recover_truncated=false, float acceptable_fraction=1.0000e+00, const std::string& dct_method="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodeAndCropJpeg", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, contents.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, crop_window.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "channels", channels);
    TFE_OpSetAttrInt(op, "ratio", ratio);
    TFE_OpSetAttrBool(op, "fancy_upscaling", (unsigned char)fancy_upscaling);
    TFE_OpSetAttrBool(op, "try_recover_truncated", (unsigned char)try_recover_truncated);
    TFE_OpSetAttrFloat(op, "acceptable_fraction", acceptable_fraction);
    TFE_OpSetAttrString(op, "dct_method", (void*) dct_method.c_str(), dct_method.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_base64(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodeBase64", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_bmp(const tensor& contents, int64_t channels=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodeBmp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, contents.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "channels", channels);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_c_s_v(const tensor& records, const std::vector<tensor>&record_defaults, const std::vector<datatype>& OUT_TYPE, const std::vector<int64_t>& select_cols, const std::string& field_delim=",", bool use_quote_delim=true, const std::string& na_value="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodeCSV", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, records.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> record_defaults_handles; record_defaults_handles.reserve((int)record_defaults.size());
    std::transform(record_defaults.begin(), record_defaults.end(), std::back_inserter(record_defaults_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, record_defaults_handles.data(), (int)record_defaults.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "OUT_TYPE", reinterpret_cast<const enum TF_DataType *>(OUT_TYPE.data()), (int)OUT_TYPE.size());
    TFE_OpSetAttrIntList(op, "select_cols", select_cols.data(), (int)select_cols.size());
    TFE_OpSetAttrString(op, "field_delim", (void*) field_delim.c_str(), (int)field_delim.size());
    TFE_OpSetAttrBool(op, "use_quote_delim", (unsigned char)use_quote_delim);
    TFE_OpSetAttrString(op, "na_value", (void*) na_value.c_str(), (int)na_value.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_compressed(const tensor& bytes, const std::string& compression_type="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodeCompressed", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, bytes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "compression_type", (void*) compression_type.c_str(), compression_type.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_gif(const tensor& contents) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodeGif", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, contents.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_j_s_o_n_example(const tensor& json_examples) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodeJSONExample", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, json_examples.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_jpeg(const tensor& contents, int64_t channels=0, int64_t ratio=1, bool fancy_upscaling=true, bool try_recover_truncated=false, float acceptable_fraction=1.0000e+00, const std::string& dct_method="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodeJpeg", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, contents.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "channels", channels);
    TFE_OpSetAttrInt(op, "ratio", ratio);
    TFE_OpSetAttrBool(op, "fancy_upscaling", (unsigned char)fancy_upscaling);
    TFE_OpSetAttrBool(op, "try_recover_truncated", (unsigned char)try_recover_truncated);
    TFE_OpSetAttrFloat(op, "acceptable_fraction", acceptable_fraction);
    TFE_OpSetAttrString(op, "dct_method", (void*) dct_method.c_str(), dct_method.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_padded_raw(const tensor& input_bytes, const tensor& fixed_length, datatype out_type, bool little_endian=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodePaddedRaw", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_bytes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, fixed_length.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);
    TFE_OpSetAttrBool(op, "little_endian", (unsigned char)little_endian);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_png(const tensor& contents, int64_t channels=0, datatype dtype=static_cast<datatype>(4)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodePng", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, contents.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "channels", channels);
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor decode_raw(const tensor& bytes, datatype out_type, bool little_endian=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DecodeRaw", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, bytes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);
    TFE_OpSetAttrBool(op, "little_endian", (unsigned char)little_endian);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor deep_copy(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DeepCopy", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dense_to_c_s_r_sparse_matrix(const tensor& dense_input, const tensor& indices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DenseToCSRSparseMatrix", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, dense_input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dense_to_sparse_batch_dataset(const tensor& input_dataset, const tensor& batch_size, const tensor& row_shape, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DenseToSparseBatchDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, batch_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, row_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor depth_to_space(const tensor& input, int64_t block_size, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DepthToSpace", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "block_size", block_size);
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor depthwise_conv2d_native(const tensor& input, const tensor& filter, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& explicit_paddings, const std::vector<int64_t>& dilations, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DepthwiseConv2dNative", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrIntList(op, "explicit_paddings", explicit_paddings.data(), (int)explicit_paddings.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor depthwise_conv2d_native_backprop_filter(const tensor& input, const tensor& filter_sizes, const tensor& out_backprop, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& explicit_paddings, const std::vector<int64_t>& dilations, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DepthwiseConv2dNativeBackpropFilter", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter_sizes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrIntList(op, "explicit_paddings", explicit_paddings.data(), (int)explicit_paddings.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor depthwise_conv2d_native_backprop_input(const tensor& input_sizes, const tensor& filter, const tensor& out_backprop, const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& explicit_paddings, const std::vector<int64_t>& dilations, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DepthwiseConv2dNativeBackpropInput", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_sizes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrIntList(op, "explicit_paddings", explicit_paddings.data(), (int)explicit_paddings.size());
    TFE_OpSetAttrIntList(op, "dilations", dilations.data(), (int)dilations.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dequantize(const tensor& input, const tensor& min_range, const tensor& max_range, const std::string& mode="MIN_COMBINED", bool narrow_range=false, int64_t axis=-1, datatype dtype=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Dequantize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, min_range.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_range.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "mode", (void*) mode.c_str(), mode.size());
    TFE_OpSetAttrBool(op, "narrow_range", (unsigned char)narrow_range);
    TFE_OpSetAttrInt(op, "axis", axis);
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor destroy_temporary_variable(const tensor& ref, const std::string& var_name) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DestroyTemporaryVariable", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "var_name", (void*) var_name.c_str(), var_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor diag(const tensor& diagonal) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Diag", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, diagonal.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor diag_part(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DiagPart", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor digamma(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Digamma", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dilation2_d(const tensor& input, const tensor& filter, const std::vector<int64_t>& strides, const std::vector<int64_t>& rates, const std::string& padding) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Dilation2D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrIntList(op, "rates", rates.data(), (int)rates.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dilation2_d_backprop_filter(const tensor& input, const tensor& filter, const tensor& out_backprop, const std::vector<int64_t>& strides, const std::vector<int64_t>& rates, const std::string& padding) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Dilation2DBackpropFilter", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrIntList(op, "rates", rates.data(), (int)rates.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dilation2_d_backprop_input(const tensor& input, const tensor& filter, const tensor& out_backprop, const std::vector<int64_t>& strides, const std::vector<int64_t>& rates, const std::string& padding) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Dilation2DBackpropInput", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrIntList(op, "rates", rates.data(), (int)rates.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor directed_interleave_dataset(const tensor& selector_input_dataset, const std::vector<tensor>&data_input_datasets, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DirectedInterleaveDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, selector_input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> data_input_datasets_handles; data_input_datasets_handles.reserve((int)data_input_datasets.size());
    std::transform(data_input_datasets.begin(), data_input_datasets.end(), std::back_inserter(data_input_datasets_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, data_input_datasets_handles.data(), (int)data_input_datasets.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "N", (int)data_input_datasets.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor div(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Div", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor div_no_nan(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DivNoNan", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor draw_bounding_boxes(const tensor& images, const tensor& boxes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DrawBoundingBoxes", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, boxes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor draw_bounding_boxes_v2(const tensor& images, const tensor& boxes, const tensor& colors) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DrawBoundingBoxesV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, boxes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, colors.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dummy_memory_cache() {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DummyMemoryCache", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dynamic_partition(const tensor& data, const tensor& partitions, int64_t num_partitions) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DynamicPartition", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, partitions.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_partitions", num_partitions);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor dynamic_stitch(const std::vector<tensor>&indices, const std::vector<tensor>&data) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "DynamicStitch", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> indices_handles; indices_handles.reserve((int)indices.size());
    std::transform(indices.begin(), indices.end(), std::back_inserter(indices_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, indices_handles.data(), (int)indices.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> data_handles; data_handles.reserve((int)data.size());
    std::transform(data.begin(), data.end(), std::back_inserter(data_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, data_handles.data(), (int)data.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", indices.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor eager_py_func(const std::vector<tensor>&input, const std::string& token, const std::vector<datatype>& Tin, const std::vector<datatype>& Tout, bool is_async=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EagerPyFunc", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> input_handles; input_handles.reserve((int)input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(input_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, input_handles.data(), (int)input.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "token", (void*) token.c_str(), token.size());
    TFE_OpSetAttrTypeList(op, "Tin", reinterpret_cast<const enum TF_DataType *>(Tin.data()), (int)Tin.size());
    TFE_OpSetAttrTypeList(op, "Tout", reinterpret_cast<const enum TF_DataType *>(Tout.data()), (int)Tout.size());
    TFE_OpSetAttrBool(op, "is_async", (unsigned char)is_async);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor edit_distance(const tensor& hypothesis_indices, const tensor& hypothesis_values, const tensor& hypothesis_shape, const tensor& truth_indices, const tensor& truth_values, const tensor& truth_shape, bool normalize=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EditDistance", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, hypothesis_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, hypothesis_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, hypothesis_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, truth_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, truth_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, truth_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "normalize", (unsigned char)normalize);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor einsum(const std::vector<tensor>&inputs, const std::string& equation) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Einsum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> inputs_handles; inputs_handles.reserve((int)inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, inputs_handles.data(), (int)inputs.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "equation", (void*) equation.c_str(), equation.size());
    TFE_OpSetAttrInt(op, "N", (int)inputs.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor elu(const tensor& features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Elu", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor elu_grad(const tensor& gradients, const tensor& outputs) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EluGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, outputs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor empty(const tensor& shape, datatype dtype, bool init=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Empty", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrBool(op, "init", (unsigned char)init);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor empty_tensor_list(const tensor& element_shape, const tensor& max_num_elements, datatype element_dtype, datatype shape_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EmptyTensorList", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, element_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_num_elements.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);
    TFE_OpSetAttrType(op, "shape_type", shape_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor encode_base64(const tensor& input, bool pad=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EncodeBase64", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "pad", (unsigned char)pad);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor encode_jpeg(const tensor& image, const std::string& format="", int64_t quality=95, bool progressive=false, bool optimize_size=false, bool chroma_downsampling=true, const std::string& density_unit="in", int64_t x_density=300, int64_t y_density=300, const std::string& xmp_metadata="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EncodeJpeg", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, image.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "format", (void*) format.c_str(), format.size());
    TFE_OpSetAttrInt(op, "quality", quality);
    TFE_OpSetAttrBool(op, "progressive", (unsigned char)progressive);
    TFE_OpSetAttrBool(op, "optimize_size", (unsigned char)optimize_size);
    TFE_OpSetAttrBool(op, "chroma_downsampling", (unsigned char)chroma_downsampling);
    TFE_OpSetAttrString(op, "density_unit", (void*) density_unit.c_str(), density_unit.size());
    TFE_OpSetAttrInt(op, "x_density", x_density);
    TFE_OpSetAttrInt(op, "y_density", y_density);
    TFE_OpSetAttrString(op, "xmp_metadata", (void*) xmp_metadata.c_str(), xmp_metadata.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor encode_jpeg_variable_quality(const tensor& images, const tensor& quality) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EncodeJpegVariableQuality", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, quality.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor encode_png(const tensor& image, int64_t compression=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EncodePng", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, image.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "compression", compression);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor encode_proto(const tensor& sizes, const std::vector<tensor>&values, const std::vector< std::string>& field_names, const std::string& message_type, const std::vector<datatype>& Tinput_types, const std::string& descriptor_source="local://") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EncodeProto", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sizes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> values_handles; values_handles.reserve(values.size());
    std::transform(values.begin(), values.end(), std::back_inserter(values_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, values_handles.data(), (int)values.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> field_names_sizes; field_names_sizes.reserve(field_names.size());
    std::transform(field_names.begin(), field_names.end(), std::back_inserter(field_names_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "field_names", reinterpret_cast<const void *const *>(field_names.data()), field_names_sizes.data(), (int)field_names.size());
    
    TFE_OpSetAttrString(op, "message_type", (void*) message_type.c_str(), message_type.size());
    TFE_OpSetAttrTypeList(op, "Tinput_types", reinterpret_cast<const enum TF_DataType *>(Tinput_types.data()), (int)Tinput_types.size());
    TFE_OpSetAttrString(op, "descriptor_source", (void*) descriptor_source.c_str(), descriptor_source.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor encode_wav(const tensor& audio, const tensor& sample_rate) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EncodeWav", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, audio.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sample_rate.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ensure_shape(const tensor& input, const std::vector<int64_t>& shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EnsureShape", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor enter(const tensor& data, const std::string& frame_name, bool is_constant=false, int64_t parallel_iterations=10) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Enter", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "frame_name", (void*) frame_name.c_str(), frame_name.size());
    TFE_OpSetAttrBool(op, "is_constant", (unsigned char)is_constant);
    TFE_OpSetAttrInt(op, "parallel_iterations", parallel_iterations);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor equal(const tensor& x, const tensor& y, bool incompatible_shape_error=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Equal", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "incompatible_shape_error", (unsigned char)incompatible_shape_error);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor erf(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Erf", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor erfc(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Erfc", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor erfinv(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Erfinv", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor euclidean_norm(const tensor& input, const tensor& reduction_indices, bool keep_dims=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "EuclideanNorm", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor exit(const tensor& data) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Exit", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor exp(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Exp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor expand_dims(const tensor& input, const tensor& dim, datatype Tdim=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExpandDims", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dim.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tdim", Tdim);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_assert_next_dataset(const tensor& input_dataset, const tensor& transformations, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalAssertNextDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, transformations.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_auto_shard_dataset(const tensor& input_dataset, const tensor& num_workers, const tensor& index, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, int64_t auto_shard_policy=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalAutoShardDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_workers.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "auto_shard_policy", auto_shard_policy);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_bytes_produced_stats_dataset(const tensor& input_dataset, const tensor& tag, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalBytesProducedStatsDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_c_s_v_dataset(const tensor& filenames, const tensor& compression_type, const tensor& buffer_size, const tensor& header, const tensor& field_delim, const tensor& use_quote_delim, const tensor& na_value, const tensor& select_cols, const std::vector<tensor>&record_defaults, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalCSVDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, filenames.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, compression_type.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, header.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, field_delim.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, use_quote_delim.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, na_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, select_cols.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> record_defaults_handles; record_defaults_handles.reserve((int)record_defaults.size());
    std::transform(record_defaults.begin(), record_defaults.end(), std::back_inserter(record_defaults_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, record_defaults_handles.data(), (int)record_defaults.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_choose_fastest_dataset(const std::vector<tensor>&input_datasets, int64_t num_experiments, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalChooseFastestDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> input_datasets_handles; input_datasets_handles.reserve((int)input_datasets.size());
    std::transform(input_datasets.begin(), input_datasets.end(), std::back_inserter(input_datasets_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, input_datasets_handles.data(), (int)input_datasets.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", (int)input_datasets.size());
    TFE_OpSetAttrInt(op, "num_experiments", num_experiments);
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_dataset_cardinality(const tensor& input_dataset) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalDatasetCardinality", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_dense_to_sparse_batch_dataset(const tensor& input_dataset, const tensor& batch_size, const tensor& row_shape, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalDenseToSparseBatchDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, batch_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, row_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_directed_interleave_dataset(const tensor& selector_input_dataset, const std::vector<tensor>&data_input_datasets, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalDirectedInterleaveDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, selector_input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> data_input_datasets_handles; data_input_datasets_handles.reserve((int)data_input_datasets.size());
    std::transform(data_input_datasets.begin(), data_input_datasets.end(), std::back_inserter(data_input_datasets_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, data_input_datasets_handles.data(), (int)data_input_datasets.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "N", (int)data_input_datasets.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_ignore_errors_dataset(const tensor& input_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalIgnoreErrorsDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_iterator_get_device(const tensor& resource) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalIteratorGetDevice", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_l_m_d_b_dataset(const tensor& filenames, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalLMDBDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, filenames.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_latency_stats_dataset(const tensor& input_dataset, const tensor& tag, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalLatencyStatsDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_matching_files_dataset(const tensor& patterns) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalMatchingFilesDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, patterns.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_max_intra_op_parallelism_dataset(const tensor& input_dataset, const tensor& max_intra_op_parallelism, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalMaxIntraOpParallelismDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_intra_op_parallelism.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_non_serializable_dataset(const tensor& input_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalNonSerializableDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_parse_example_dataset(const tensor& input_dataset, const tensor& num_parallel_calls, const std::vector<tensor>&dense_defaults, const std::vector< std::string>& sparse_keys, const std::vector< std::string>& dense_keys, const std::vector<datatype>& sparse_types, const std::vector<datatype>& Tdense, const std::vector< std::vector<int64_t>>& dense_shapes, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, bool sloppy=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalParseExampleDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_parallel_calls.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> dense_defaults_handles; dense_defaults_handles.reserve(dense_defaults.size());
    std::transform(dense_defaults.begin(), dense_defaults.end(), std::back_inserter(dense_defaults_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, dense_defaults_handles.data(), (int)dense_defaults.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> sparse_keys_sizes; sparse_keys_sizes.reserve((int)sparse_keys.size());
    std::transform(sparse_keys.begin(), sparse_keys.end(), std::back_inserter(sparse_keys_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "sparse_keys", reinterpret_cast<const void *const *>(sparse_keys.data()), sparse_keys_sizes.data(), (int)sparse_keys.size());
    
    
    std::vector<std::size_t> dense_keys_sizes; dense_keys_sizes.reserve(dense_keys.size());
    std::transform(dense_keys.begin(), dense_keys.end(), std::back_inserter(dense_keys_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "dense_keys", reinterpret_cast<const void *const *>(dense_keys.data()), dense_keys_sizes.data(), (int)dense_keys.size());
    
    TFE_OpSetAttrTypeList(op, "sparse_types", reinterpret_cast<const enum TF_DataType *>(sparse_types.data()), (int)sparse_types.size());
    TFE_OpSetAttrTypeList(op, "Tdense", reinterpret_cast<const enum TF_DataType *>(Tdense.data()), (int)Tdense.size());
    
    std::vector<const int64_t*> dense_shapes_values; dense_shapes_values.reserve((int)dense_shapes.size());
    std::vector<int> dense_shapes_ndims; dense_shapes_ndims.reserve((int)dense_shapes.size());
    std::transform(dense_shapes.begin(), dense_shapes.end(), std::back_inserter(dense_shapes_values), [](const auto& v) { return v.data();});
    std::transform(dense_shapes.begin(), dense_shapes.end(), std::back_inserter(dense_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "dense_shapes", dense_shapes_values.data(), dense_shapes_ndims.data(), (int)dense_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrBool(op, "sloppy", (unsigned char)sloppy);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_private_thread_pool_dataset(const tensor& input_dataset, const tensor& num_threads, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalPrivateThreadPoolDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_threads.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_random_dataset(const tensor& seed, const tensor& seed2, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalRandomDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_rebatch_dataset(const tensor& input_dataset, const tensor& num_replicas, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, bool use_fallback=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalRebatchDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_replicas.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrBool(op, "use_fallback", (unsigned char)use_fallback);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_set_stats_aggregator_dataset(const tensor& input_dataset, const tensor& stats_aggregator, const tensor& tag, const tensor& counter_prefix, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalSetStatsAggregatorDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, stats_aggregator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, counter_prefix.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_sleep_dataset(const tensor& input_dataset, const tensor& sleep_microseconds, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalSleepDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sleep_microseconds.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_sliding_window_dataset(const tensor& input_dataset, const tensor& window_size, const tensor& window_shift, const tensor& window_stride, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalSlidingWindowDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, window_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, window_shift.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, window_stride.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_sql_dataset(const tensor& driver_name, const tensor& data_source_name, const tensor& query, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalSqlDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, driver_name.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, data_source_name.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, query.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_stats_aggregator_handle(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalStatsAggregatorHandle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_stats_aggregator_summary(const tensor& iterator) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalStatsAggregatorSummary", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, iterator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_thread_pool_dataset(const tensor& input_dataset, const tensor& thread_pool, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalThreadPoolDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, thread_pool.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_thread_pool_handle(int64_t num_threads, const std::string& display_name, int64_t max_intra_op_parallelism=1, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalThreadPoolHandle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_threads", num_threads);
    TFE_OpSetAttrString(op, "display_name", (void*) display_name.c_str(), display_name.size());
    TFE_OpSetAttrInt(op, "max_intra_op_parallelism", max_intra_op_parallelism);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_unbatch_dataset(const tensor& input_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalUnbatchDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor experimental_unique_dataset(const tensor& input_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExperimentalUniqueDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor expint(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Expint", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor expm1(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Expm1", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor extract_glimpse(const tensor& input, const tensor& size, const tensor& offsets, bool centered=true, bool normalized=true, bool uniform_noise=true, const std::string& noise="uniform") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExtractGlimpse", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, offsets.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "centered", (unsigned char)centered);
    TFE_OpSetAttrBool(op, "normalized", (unsigned char)normalized);
    TFE_OpSetAttrBool(op, "uniform_noise", (unsigned char)uniform_noise);
    TFE_OpSetAttrString(op, "noise", (void*) noise.c_str(), noise.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor extract_image_patches(const tensor& images, const std::vector<int64_t>& ksizes, const std::vector<int64_t>& strides, const std::vector<int64_t>& rates, const std::string& padding) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExtractImagePatches", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksizes", ksizes.data(), (int)ksizes.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrIntList(op, "rates", rates.data(), (int)rates.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor extract_jpeg_shape(const tensor& contents, datatype output_type=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExtractJpegShape", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, contents.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "output_type", output_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor extract_volume_patches(const tensor& input, const std::vector<int64_t>& ksizes, const std::vector<int64_t>& strides, const std::string& padding) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ExtractVolumePatches", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksizes", ksizes.data(), (int)ksizes.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor f_f_t(const tensor& input, datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FFT", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor f_f_t2_d(const tensor& input, datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FFT2D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor f_f_t3_d(const tensor& input, datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FFT3D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor f_i_f_o_queue(const std::vector<datatype>& component_types, const std::vector< std::vector<int64_t>>& shapes, int64_t capacity=-1, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FIFOQueue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor f_i_f_o_queue_v2(const std::vector<datatype>& component_types, const std::vector< std::vector<int64_t>>& shapes, int64_t capacity=-1, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FIFOQueueV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fact() {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Fact", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fake_param(datatype dtype, const std::vector<int64_t>& shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FakeParam", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fake_quant_with_min_max_args(const tensor& inputs, float min=-6.0000e+00, float max=6.0000e+00, int64_t num_bits=8, bool narrow_range=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FakeQuantWithMinMaxArgs", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, inputs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrFloat(op, "min", min);
    TFE_OpSetAttrFloat(op, "max", max);
    TFE_OpSetAttrInt(op, "num_bits", num_bits);
    TFE_OpSetAttrBool(op, "narrow_range", (unsigned char)narrow_range);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fake_quant_with_min_max_args_gradient(const tensor& gradients, const tensor& inputs, float min=-6.0000e+00, float max=6.0000e+00, int64_t num_bits=8, bool narrow_range=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FakeQuantWithMinMaxArgsGradient", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, inputs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrFloat(op, "min", min);
    TFE_OpSetAttrFloat(op, "max", max);
    TFE_OpSetAttrInt(op, "num_bits", num_bits);
    TFE_OpSetAttrBool(op, "narrow_range", (unsigned char)narrow_range);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fake_quant_with_min_max_vars(const tensor& inputs, const tensor& min, const tensor& max, int64_t num_bits=8, bool narrow_range=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FakeQuantWithMinMaxVars", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, inputs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, min.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_bits", num_bits);
    TFE_OpSetAttrBool(op, "narrow_range", (unsigned char)narrow_range);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fake_quant_with_min_max_vars_per_channel(const tensor& inputs, const tensor& min, const tensor& max, int64_t num_bits=8, bool narrow_range=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FakeQuantWithMinMaxVarsPerChannel", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, inputs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, min.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_bits", num_bits);
    TFE_OpSetAttrBool(op, "narrow_range", (unsigned char)narrow_range);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fake_queue(const tensor& resource) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FakeQueue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fill(const tensor& dims, const tensor& value, datatype index_type=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Fill", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, dims.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "index_type", index_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor filter_by_last_component_dataset(const tensor& input_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FilterByLastComponentDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fingerprint(const tensor& data, const tensor& method) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Fingerprint", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, method.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fixed_length_record_dataset(const tensor& filenames, const tensor& header_bytes, const tensor& record_bytes, const tensor& footer_bytes, const tensor& buffer_size) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FixedLengthRecordDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, filenames.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, header_bytes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, record_bytes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, footer_bytes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fixed_length_record_dataset_v2(const tensor& filenames, const tensor& header_bytes, const tensor& record_bytes, const tensor& footer_bytes, const tensor& buffer_size, const tensor& compression_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FixedLengthRecordDatasetV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, filenames.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, header_bytes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, record_bytes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, footer_bytes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, compression_type.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fixed_length_record_reader(int64_t record_bytes, int64_t header_bytes=0, int64_t footer_bytes=0, int64_t hop_bytes=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FixedLengthRecordReader", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrInt(op, "record_bytes", record_bytes);
    TFE_OpSetAttrInt(op, "header_bytes", header_bytes);
    TFE_OpSetAttrInt(op, "footer_bytes", footer_bytes);
    TFE_OpSetAttrInt(op, "hop_bytes", hop_bytes);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fixed_length_record_reader_v2(int64_t record_bytes, int64_t header_bytes=0, int64_t footer_bytes=0, int64_t hop_bytes=0, const std::string& container="", const std::string& shared_name="", const std::string& encoding="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FixedLengthRecordReaderV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrInt(op, "record_bytes", record_bytes);
    TFE_OpSetAttrInt(op, "header_bytes", header_bytes);
    TFE_OpSetAttrInt(op, "footer_bytes", footer_bytes);
    TFE_OpSetAttrInt(op, "hop_bytes", hop_bytes);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrString(op, "encoding", (void*) encoding.c_str(), encoding.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor floor(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Floor", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor floor_div(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FloorDiv", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor floor_mod(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FloorMod", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fractional_avg_pool_grad(const tensor& orig_input_input_tensor_shape, const tensor& out_backprop, const tensor& row_pooling_sequence, const tensor& col_pooling_sequence, bool overlapping=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FractionalAvgPoolGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, orig_input_input_tensor_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, row_pooling_sequence.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, col_pooling_sequence.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "overlapping", (unsigned char)overlapping);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fractional_max_pool_grad(const tensor& orig_input, const tensor& orig_output, const tensor& out_backprop, const tensor& row_pooling_sequence, const tensor& col_pooling_sequence, bool overlapping=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FractionalMaxPoolGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, orig_input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, orig_output.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, out_backprop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, row_pooling_sequence.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, col_pooling_sequence.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "overlapping", (unsigned char)overlapping);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fresnel_cos(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FresnelCos", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fresnel_sin(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FresnelSin", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fused_pad_conv2_d(const tensor& input, const tensor& paddings, const tensor& filter, const std::string& mode, const std::vector<int64_t>& strides, const std::string& padding) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FusedPadConv2D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, paddings.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "mode", (void*) mode.c_str(), mode.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor fused_resize_and_pad_conv2_d(const tensor& input, const tensor& size, const tensor& paddings, const tensor& filter, const std::string& mode, const std::vector<int64_t>& strides, const std::string& padding, bool resize_align_corners=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "FusedResizeAndPadConv2D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, paddings.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, filter.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "mode", (void*) mode.c_str(), mode.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());
    TFE_OpSetAttrBool(op, "resize_align_corners", (unsigned char)resize_align_corners);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor gather(const tensor& params, const tensor& indices, datatype Tparams, datatype Tindices, bool validate_indices=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Gather", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, params.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tparams", Tparams);
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "validate_indices", (unsigned char)validate_indices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor gather_nd(const tensor& params, const tensor& indices, datatype Tparams, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "GatherNd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, params.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tparams", Tparams);
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor gather_v2(const tensor& params, const tensor& indices, const tensor& axis, datatype Tparams, datatype Tindices, datatype Taxis, int64_t batch_dims=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "GatherV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, params.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, axis.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tparams", Tparams);
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrType(op, "Taxis", Taxis);
    TFE_OpSetAttrInt(op, "batch_dims", batch_dims);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor get_session_handle(const tensor& value) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "GetSessionHandle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor get_session_handle_v2(const tensor& value) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "GetSessionHandleV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor get_session_tensor(const tensor& handle, datatype dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "GetSessionTensor", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor greater(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Greater", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor greater_equal(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "GreaterEqual", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor guarantee_const_tensor(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "GuaranteeConst", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor h_s_v_to_r_g_b(const tensor& images) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "HSVToRGB", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor hash_table(datatype key_dtype, datatype value_dtype, const std::string& container="", const std::string& shared_name="", bool use_node_name_sharing=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "HashTable", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "key_dtype", key_dtype);
    TFE_OpSetAttrType(op, "value_dtype", value_dtype);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrBool(op, "use_node_name_sharing", (unsigned char)use_node_name_sharing);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor hash_table_v2(datatype key_dtype, datatype value_dtype, const std::string& container="", const std::string& shared_name="", bool use_node_name_sharing=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "HashTableV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "key_dtype", key_dtype);
    TFE_OpSetAttrType(op, "value_dtype", value_dtype);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrBool(op, "use_node_name_sharing", (unsigned char)use_node_name_sharing);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor histogram_fixed_width(const tensor& values, const tensor& value_range, const tensor& nbins, datatype dtype=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "HistogramFixedWidth", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value_range.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, nbins.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor histogram_summary(const tensor& tag, const tensor& values) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "HistogramSummary", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor i_f_f_t(const tensor& input, datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IFFT", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor i_f_f_t2_d(const tensor& input, datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IFFT2D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor i_f_f_t3_d(const tensor& input, datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IFFT3D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor i_r_f_f_t(const tensor& input, const tensor& fft_length, datatype Treal=static_cast<datatype>(1), datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IRFFT", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, fft_length.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Treal", Treal);
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor i_r_f_f_t2_d(const tensor& input, const tensor& fft_length, datatype Treal=static_cast<datatype>(1), datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IRFFT2D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, fft_length.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Treal", Treal);
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor i_r_f_f_t3_d(const tensor& input, const tensor& fft_length, datatype Treal=static_cast<datatype>(1), datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IRFFT3D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, fft_length.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Treal", Treal);
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor identity(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Identity", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor identity_n(const std::vector<tensor>&input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IdentityN", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> input_handles; input_handles.reserve((int)input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(input_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, input_handles.data(), (int)input.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor identity_reader(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IdentityReader", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor identity_reader_v2(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IdentityReaderV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor igamma(const tensor& a, const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Igamma", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor igamma_grad_a(const tensor& a, const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IgammaGradA", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor igammac(const tensor& a, const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Igammac", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ignore_errors_dataset(const tensor& input_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IgnoreErrorsDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor imag(const tensor& input, datatype Tout=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Imag", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tout", Tout);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor image_projective_transform_v2(const tensor& images, const tensor& transforms, const tensor& output_shape, datatype dtype, const std::string& interpolation, const std::string& fill_mode="CONSTANT") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ImageProjectiveTransformV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, transforms.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, output_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrString(op, "interpolation", (void*) interpolation.c_str(), interpolation.size());
    TFE_OpSetAttrString(op, "fill_mode", (void*) fill_mode.c_str(), fill_mode.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor image_summary(const tensor& tag, const tensor& input_tensor, const tensor& bad_color, int64_t max_images=3) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ImageSummary", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    TFE_OpSetAttrTensor(op, "bad_color", bad_color.tf_tensor.get(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "max_images", max_images);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor immutable_const_tensor(datatype dtype, const std::vector<int64_t>& shape, const std::string& memory_region_name) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ImmutableConst", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "memory_region_name", (void*) memory_region_name.c_str(), memory_region_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor in_top_k(const tensor& predictions, const tensor& targets, int64_t k) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "InTopK", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, predictions.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, targets.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "k", k);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor in_top_k_v2(const tensor& predictions, const tensor& targets, const tensor& k) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "InTopKV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, predictions.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, targets.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, k.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor infeed_dequeue(datatype dtype, const std::vector<int64_t>& shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "InfeedDequeue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor infeed_dequeue_tuple(const std::vector<datatype>& dtypes, const std::vector< std::vector<int64_t>>& shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "InfeedDequeueTuple", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor inplace_add(const tensor& x, const tensor& i, const tensor& v) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "InplaceAdd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, i.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, v.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor inplace_sub(const tensor& x, const tensor& i, const tensor& v) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "InplaceSub", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, i.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, v.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor inplace_update(const tensor& x, const tensor& i, const tensor& v) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "InplaceUpdate", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, i.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, v.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor inv(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Inv", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor inv_grad(const tensor& y, const tensor& dy) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "InvGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dy.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor invert(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Invert", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor invert_permutation(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "InvertPermutation", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor is_boosted_trees_ensemble_initialized(const tensor& tree_ensemble_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IsBoostedTreesEnsembleInitialized", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, tree_ensemble_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor is_boosted_trees_quantile_stream_resource_initialized(const tensor& quantile_stream_resource_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IsBoostedTreesQuantileStreamResourceInitialized", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, quantile_stream_resource_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor is_finite(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IsFinite", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor is_inf(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IsInf", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor is_nan(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IsNan", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor is_variable_initialized(const tensor& ref, datatype dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IsVariableInitialized", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor iterator(const std::string& shared_name, const std::string& container, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Iterator", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor iterator_from_string_handle(const tensor& string_handle, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IteratorFromStringHandle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, string_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor iterator_from_string_handle_v2(const tensor& string_handle, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IteratorFromStringHandleV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, string_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor iterator_get_device(const tensor& resource) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IteratorGetDevice", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor iterator_get_next(const tensor& iterator, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IteratorGetNext", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, iterator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor iterator_get_next_as_optional(const tensor& iterator, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IteratorGetNextAsOptional", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, iterator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor iterator_get_next_sync(const tensor& iterator, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IteratorGetNextSync", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, iterator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor iterator_to_string_handle(const tensor& resource_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IteratorToStringHandle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor iterator_v2(const std::string& shared_name, const std::string& container, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "IteratorV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor l2_loss(const tensor& t) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "L2Loss", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, t.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor l_m_d_b_dataset(const tensor& filenames, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LMDBDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, filenames.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor l_m_d_b_reader(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LMDBReader", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor l_r_n(const tensor& input, int64_t depth_radius=5, float bias=1.0000e+00, float alpha=1.0000e+00, float beta=5.0000e-01) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LRN", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "depth_radius", depth_radius);
    TFE_OpSetAttrFloat(op, "bias", bias);
    TFE_OpSetAttrFloat(op, "alpha", alpha);
    TFE_OpSetAttrFloat(op, "beta", beta);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor l_r_n_grad(const tensor& input_grads, const tensor& input_image, const tensor& output_image, int64_t depth_radius=5, float bias=1.0000e+00, float alpha=1.0000e+00, float beta=5.0000e-01) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LRNGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_grads.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_image.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, output_image.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "depth_radius", depth_radius);
    TFE_OpSetAttrFloat(op, "bias", bias);
    TFE_OpSetAttrFloat(op, "alpha", alpha);
    TFE_OpSetAttrFloat(op, "beta", beta);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor latency_stats_dataset(const tensor& input_dataset, const tensor& tag, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LatencyStatsDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor leaky_relu(const tensor& features, float alpha=2.0000e-01) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LeakyRelu", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrFloat(op, "alpha", alpha);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor leaky_relu_grad(const tensor& gradients, const tensor& features, float alpha=2.0000e-01) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LeakyReluGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrFloat(op, "alpha", alpha);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor left_shift(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LeftShift", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor less(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Less", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor less_equal(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LessEqual", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor lgamma(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Lgamma", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor lin_space(const tensor& start, const tensor& stop, const tensor& num, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LinSpace", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, start.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, stop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor load_and_remap_matrix(const tensor& ckpt_path, const tensor& old_input_tensor_name, const tensor& row_remapping, const tensor& col_remapping, const tensor& initializing_values, int64_t num_rows, int64_t num_cols, int64_t max_rows_in_memory=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LoadAndRemapMatrix", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ckpt_path.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, old_input_tensor_name.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, row_remapping.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, col_remapping.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, initializing_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_rows", num_rows);
    TFE_OpSetAttrInt(op, "num_cols", num_cols);
    TFE_OpSetAttrInt(op, "max_rows_in_memory", max_rows_in_memory);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor log(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Log", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor log1p(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Log1p", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor log_softmax(const tensor& logits) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LogSoftmax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, logits.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor logical_and(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LogicalAnd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor logical_not(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LogicalNot", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor logical_or(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LogicalOr", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor lookup_table_find(const tensor& table_handle, const tensor& keys, const tensor& default_value, datatype Tin, datatype Tout) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LookupTableFind", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, table_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, keys.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, default_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tin", Tin);
    TFE_OpSetAttrType(op, "Tout", Tout);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor lookup_table_find_v2(const tensor& table_handle, const tensor& keys, const tensor& default_value, datatype Tin, datatype Tout) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LookupTableFindV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, table_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, keys.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, default_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tin", Tin);
    TFE_OpSetAttrType(op, "Tout", Tout);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor lookup_table_size(const tensor& table_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LookupTableSize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, table_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor lookup_table_size_v2(const tensor& table_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LookupTableSizeV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, table_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor loop_cond(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LoopCond", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor lower_bound(const tensor& sorted_inputs, const tensor& values, datatype out_type=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "LowerBound", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sorted_inputs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor map_incomplete_size(const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MapIncompleteSize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor map_peek(const tensor& key, const tensor& indices, const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MapPeek", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, key.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor map_size(const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MapSize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor map_unstage(const tensor& key, const tensor& indices, const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MapUnstage", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, key.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mat_mul(const tensor& a, const tensor& b, bool transpose_a=false, bool transpose_b=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatMul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "transpose_a", (unsigned char)transpose_a);
    TFE_OpSetAttrBool(op, "transpose_b", (unsigned char)transpose_b);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matching_files(const tensor& pattern) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatchingFiles", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, pattern.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matching_files_dataset(const tensor& patterns) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatchingFilesDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, patterns.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_band_part(const tensor& input, const tensor& num_lower, const tensor& num_upper, datatype Tindex=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixBandPart", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_lower.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_upper.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindex", Tindex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_determinant(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixDeterminant", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_diag(const tensor& diagonal) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixDiag", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, diagonal.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_diag_part(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixDiagPart", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_diag_part_v2(const tensor& input, const tensor& k, const tensor& padding_value) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixDiagPartV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, k.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, padding_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_diag_part_v3(const tensor& input, const tensor& k, const tensor& padding_value, const std::string& align="RIGHT_LEFT") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixDiagPartV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, k.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, padding_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "align", (void*) align.c_str(), align.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_diag_v2(const tensor& diagonal, const tensor& k, const tensor& num_rows, const tensor& num_cols, const tensor& padding_value) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixDiagV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, diagonal.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, k.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_rows.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_cols.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, padding_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_diag_v3(const tensor& diagonal, const tensor& k, const tensor& num_rows, const tensor& num_cols, const tensor& padding_value, const std::string& align="RIGHT_LEFT") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixDiagV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, diagonal.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, k.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_rows.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_cols.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, padding_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "align", (void*) align.c_str(), align.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_exponential(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixExponential", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_inverse(const tensor& input, bool adjoint=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixInverse", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "adjoint", (unsigned char)adjoint);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_logarithm(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixLogarithm", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_set_diag(const tensor& input, const tensor& diagonal) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixSetDiag", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, diagonal.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_set_diag_v2(const tensor& input, const tensor& diagonal, const tensor& k) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixSetDiagV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, diagonal.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, k.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_set_diag_v3(const tensor& input, const tensor& diagonal, const tensor& k, const std::string& align="RIGHT_LEFT") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixSetDiagV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, diagonal.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, k.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "align", (void*) align.c_str(), align.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_solve(const tensor& matrix, const tensor& rhs, bool adjoint=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixSolve", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, matrix.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rhs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "adjoint", (unsigned char)adjoint);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_solve_ls(const tensor& matrix, const tensor& rhs, const tensor& l2_regularizer, bool fast=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixSolveLs", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, matrix.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rhs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2_regularizer.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "fast", (unsigned char)fast);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_square_root(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixSquareRoot", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor matrix_triangular_solve(const tensor& matrix, const tensor& rhs, bool lower=true, bool adjoint=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MatrixTriangularSolve", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, matrix.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rhs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "lower", (unsigned char)lower);
    TFE_OpSetAttrBool(op, "adjoint", (unsigned char)adjoint);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max(const tensor& input, const tensor& reduction_indices, bool keep_dims=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Max", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_intra_op_parallelism_dataset(const tensor& input_dataset, const tensor& max_intra_op_parallelism, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxIntraOpParallelismDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_intra_op_parallelism.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool(const tensor& input, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPool", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool3_d(const tensor& input, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, const std::string& data_format="NDHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPool3D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool3_d_grad(const tensor& orig_input, const tensor& orig_output, const tensor& grad, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, const std::string& data_format="NDHWC", datatype TInput=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPool3DGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, orig_input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, orig_output.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());
    TFE_OpSetAttrType(op, "TInput", TInput);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool3_d_grad_grad(const tensor& orig_input, const tensor& orig_output, const tensor& grad, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, const std::string& data_format="NDHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPool3DGradGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, orig_input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, orig_output.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool_grad(const tensor& orig_input, const tensor& orig_output, const tensor& grad, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPoolGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, orig_input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, orig_output.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool_grad_grad(const tensor& orig_input, const tensor& orig_output, const tensor& grad, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPoolGradGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, orig_input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, orig_output.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), (int)data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool_grad_grad_v2(const tensor& orig_input, const tensor& orig_output, const tensor& grad, const tensor& ksize, const tensor& strides, const std::string& padding, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPoolGradGradV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, orig_input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, orig_output.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, ksize.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, strides.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool_grad_grad_with_argmax(const tensor& input, const tensor& grad, const tensor& argmax, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, datatype Targmax, bool include_batch_in_index=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPoolGradGradWithArgmax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, argmax.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrType(op, "Targmax", Targmax);
    TFE_OpSetAttrBool(op, "include_batch_in_index", (unsigned char)include_batch_in_index);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool_grad_v2(const tensor& orig_input, const tensor& orig_output, const tensor& grad, const tensor& ksize, const tensor& strides, const std::string& padding, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPoolGradV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, orig_input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, orig_output.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, ksize.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, strides.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool_grad_with_argmax(const tensor& input, const tensor& grad, const tensor& argmax, const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides, const std::string& padding, datatype Targmax, bool include_batch_in_index=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPoolGradWithArgmax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, argmax.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "ksize", ksize.data(), (int)ksize.size());
    TFE_OpSetAttrIntList(op, "strides", strides.data(), (int)strides.size());
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), (int)padding.size());
    TFE_OpSetAttrType(op, "Targmax", Targmax);
    TFE_OpSetAttrBool(op, "include_batch_in_index", (unsigned char)include_batch_in_index);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor max_pool_v2(const tensor& input, const tensor& ksize, const tensor& strides, const std::string& padding, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MaxPoolV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, ksize.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, strides.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "padding", (void*) padding.c_str(), padding.size());
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor maximum(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Maximum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mean(const tensor& input, const tensor& reduction_indices, bool keep_dims=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Mean", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor merge_summary(const std::vector<tensor>&inputs) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MergeSummary", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> inputs_handles; inputs_handles.reserve((int)inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, inputs_handles.data(), (int)inputs.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", (int)inputs.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mfcc(const tensor& spectrogram, const tensor& sample_rate, float upper_frequency_limit=4.0000e+03, float lower_frequency_limit=2.0000e+01, int64_t filterbank_channel_count=40, int64_t dct_coefficient_count=13) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Mfcc", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, spectrogram.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sample_rate.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrFloat(op, "upper_frequency_limit", upper_frequency_limit);
    TFE_OpSetAttrFloat(op, "lower_frequency_limit", lower_frequency_limit);
    TFE_OpSetAttrInt(op, "filterbank_channel_count", filterbank_channel_count);
    TFE_OpSetAttrInt(op, "dct_coefficient_count", dct_coefficient_count);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor min(const tensor& input, const tensor& reduction_indices, bool keep_dims=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Min", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor minimum(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Minimum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mirror_pad(const tensor& input, const tensor& paddings, const std::string& mode, datatype Tpaddings=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MirrorPad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, paddings.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "mode", (void*) mode.c_str(), mode.size());
    TFE_OpSetAttrType(op, "Tpaddings", Tpaddings);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mirror_pad_grad(const tensor& input, const tensor& paddings, const std::string& mode, datatype Tpaddings=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MirrorPadGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, paddings.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "mode", (void*) mode.c_str(), mode.size());
    TFE_OpSetAttrType(op, "Tpaddings", Tpaddings);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mod(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Mod", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor model_dataset(const tensor& input_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, int64_t algorithm=0, int64_t cpu_budget=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ModelDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "algorithm", algorithm);
    TFE_OpSetAttrInt(op, "cpu_budget", cpu_budget);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mul(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Mul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mul_no_nan(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MulNoNan", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor multi_device_iterator(const std::vector< std::string>& devices, const std::string& shared_name, const std::string& container, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MultiDeviceIterator", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    
    std::vector<std::size_t> devices_sizes; devices_sizes.reserve(devices.size());
    std::transform(devices.begin(), devices.end(), std::back_inserter(devices_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "devices", reinterpret_cast<const void *const *>(devices.data()), devices_sizes.data(), (int)devices.size());
    
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor multi_device_iterator_from_string_handle(const tensor& string_handle, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MultiDeviceIteratorFromStringHandle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, string_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor multi_device_iterator_get_next_from_shard(const tensor& multi_device_iterator, const tensor& shard_num, const tensor& incarnation_id, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MultiDeviceIteratorGetNextFromShard", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, multi_device_iterator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shard_num.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, incarnation_id.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor multi_device_iterator_init(const tensor& dataset, const tensor& multi_device_iterator, const tensor& max_buffer_size) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MultiDeviceIteratorInit", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, multi_device_iterator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor multi_device_iterator_to_string_handle(const tensor& multi_device_iterator) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MultiDeviceIteratorToStringHandle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, multi_device_iterator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor multinomial(const tensor& logits, const tensor& num_samples, int64_t seed=0, int64_t seed2=0, datatype output_dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Multinomial", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, logits.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_samples.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);
    TFE_OpSetAttrType(op, "output_dtype", output_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mutable_dense_hash_table(const tensor& empty_key, datatype key_dtype, datatype value_dtype, const std::vector<int64_t>& value_shape, const std::string& container="", const std::string& shared_name="", bool use_node_name_sharing=false, int64_t initial_num_buckets=131072, float max_load_factor=8.0000e-01) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MutableDenseHashTable", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, empty_key.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "key_dtype", key_dtype);
    TFE_OpSetAttrType(op, "value_dtype", value_dtype);
    
    TFE_OpSetAttrShape(op, "value_shape", value_shape.data(), (int)value_shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrBool(op, "use_node_name_sharing", (unsigned char)use_node_name_sharing);
    TFE_OpSetAttrInt(op, "initial_num_buckets", initial_num_buckets);
    TFE_OpSetAttrFloat(op, "max_load_factor", max_load_factor);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mutable_dense_hash_table_v2(const tensor& empty_key, const tensor& deleted_key, datatype key_dtype, datatype value_dtype, const std::vector<int64_t>& value_shape, const std::string& container="", const std::string& shared_name="", bool use_node_name_sharing=false, int64_t initial_num_buckets=131072, float max_load_factor=8.0000e-01) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MutableDenseHashTableV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, empty_key.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, deleted_key.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "key_dtype", key_dtype);
    TFE_OpSetAttrType(op, "value_dtype", value_dtype);
    
    TFE_OpSetAttrShape(op, "value_shape", value_shape.data(), (int)value_shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrBool(op, "use_node_name_sharing", (unsigned char)use_node_name_sharing);
    TFE_OpSetAttrInt(op, "initial_num_buckets", initial_num_buckets);
    TFE_OpSetAttrFloat(op, "max_load_factor", max_load_factor);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mutable_hash_table(datatype key_dtype, datatype value_dtype, const std::string& container="", const std::string& shared_name="", bool use_node_name_sharing=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MutableHashTable", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "key_dtype", key_dtype);
    TFE_OpSetAttrType(op, "value_dtype", value_dtype);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrBool(op, "use_node_name_sharing", (unsigned char)use_node_name_sharing);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mutable_hash_table_of_tensors(datatype key_dtype, datatype value_dtype, const std::vector<int64_t>& value_shape, const std::string& container="", const std::string& shared_name="", bool use_node_name_sharing=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MutableHashTableOfTensors", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "key_dtype", key_dtype);
    TFE_OpSetAttrType(op, "value_dtype", value_dtype);
    
    TFE_OpSetAttrShape(op, "value_shape", value_shape.data(), (int)value_shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrBool(op, "use_node_name_sharing", (unsigned char)use_node_name_sharing);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mutable_hash_table_of_tensors_v2(datatype key_dtype, datatype value_dtype, const std::vector<int64_t>& value_shape, const std::string& container="", const std::string& shared_name="", bool use_node_name_sharing=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MutableHashTableOfTensorsV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "key_dtype", key_dtype);
    TFE_OpSetAttrType(op, "value_dtype", value_dtype);
    
    TFE_OpSetAttrShape(op, "value_shape", value_shape.data(), (int)value_shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrBool(op, "use_node_name_sharing", (unsigned char)use_node_name_sharing);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mutable_hash_table_v2(datatype key_dtype, datatype value_dtype, const std::string& container="", const std::string& shared_name="", bool use_node_name_sharing=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MutableHashTableV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "key_dtype", key_dtype);
    TFE_OpSetAttrType(op, "value_dtype", value_dtype);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrBool(op, "use_node_name_sharing", (unsigned char)use_node_name_sharing);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mutex_lock(const tensor& mutex) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MutexLock", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, mutex.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor mutex_v2(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "MutexV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor nccl_all_reduce(const tensor& input, const std::string& reduction, int64_t num_devices, const std::string& shared_name) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NcclAllReduce", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "reduction", (void*) reduction.c_str(), reduction.size());
    TFE_OpSetAttrInt(op, "num_devices", num_devices);
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor nccl_broadcast(const tensor& input, const std::vector<int64_t>& shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NcclBroadcast", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor nccl_reduce(const std::vector<tensor>&input, const std::string& reduction) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NcclReduce", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> input_handles; input_handles.reserve((int)input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(input_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, input_handles.data(), (int)input.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "reduction", (void*) reduction.c_str(), reduction.size());
    TFE_OpSetAttrInt(op, "num_devices", (int)input.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ndtri(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Ndtri", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor neg(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Neg", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor next_after(const tensor& x1, const tensor& x2) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NextAfter", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, x2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor next_iteration(const tensor& data) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NextIteration", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor non_deterministic_ints(const tensor& shape, datatype dtype=static_cast<datatype>(9), datatype shape_dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NonDeterministicInts", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "shape_dtype", shape_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor non_max_suppression(const tensor& boxes, const tensor& scores, const tensor& max_output_size, float iou_threshold=5.0000e-01) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NonMaxSuppression", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, boxes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, scores.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_output_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrFloat(op, "iou_threshold", iou_threshold);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor non_max_suppression_v2(const tensor& boxes, const tensor& scores, const tensor& max_output_size, const tensor& iou_threshold, datatype T_threshold=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NonMaxSuppressionV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, boxes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, scores.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_output_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, iou_threshold.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "T_threshold", T_threshold);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor non_max_suppression_v3(const tensor& boxes, const tensor& scores, const tensor& max_output_size, const tensor& iou_threshold, const tensor& score_threshold, datatype T_threshold=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NonMaxSuppressionV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, boxes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, scores.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_output_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, iou_threshold.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, score_threshold.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "T_threshold", T_threshold);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor non_max_suppression_with_overlaps(const tensor& overlaps, const tensor& scores, const tensor& max_output_size, const tensor& overlap_threshold, const tensor& score_threshold) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NonMaxSuppressionWithOverlaps", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, overlaps.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, scores.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_output_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, overlap_threshold.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, score_threshold.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor non_serializable_dataset(const tensor& input_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NonSerializableDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor not_equal(const tensor& x, const tensor& y, bool incompatible_shape_error=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NotEqual", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "incompatible_shape_error", (unsigned char)incompatible_shape_error);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor nth_element(const tensor& input, const tensor& n, bool reverse=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "NthElement", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, n.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "reverse", (unsigned char)reverse);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor one_hot(const tensor& indices, const tensor& depth, const tensor& on_value, const tensor& off_value, int64_t axis=-1, datatype TI=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OneHot", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, depth.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, on_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, off_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "axis", axis);
    TFE_OpSetAttrType(op, "TI", TI);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ones_like(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OnesLike", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor optimize_dataset(const tensor& input_dataset, const tensor& optimizations, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, const std::vector< std::string>& optimization_configs) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OptimizeDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, optimizations.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<std::size_t> optimization_configs_sizes; optimization_configs_sizes.reserve(optimization_configs.size());
    std::transform(optimization_configs.begin(), optimization_configs.end(), std::back_inserter(optimization_configs_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "optimization_configs", reinterpret_cast<const void *const *>(optimization_configs.data()), optimization_configs_sizes.data(), (int)optimization_configs.size());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor optional_from_value(const std::vector<tensor>&components, const std::vector<datatype>& Toutput_types) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OptionalFromValue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> components_handles; components_handles.reserve((int)components.size());
    std::transform(components.begin(), components.end(), std::back_inserter(components_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, components_handles.data(), (int)components.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "Toutput_types", reinterpret_cast<const enum TF_DataType *>(Toutput_types.data()), (int)Toutput_types.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor optional_get_value(const tensor& optional, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OptionalGetValue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, optional.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor optional_has_value(const tensor& optional) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OptionalHasValue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, optional.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor optional_none() {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OptionalNone", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ordered_map_incomplete_size(const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OrderedMapIncompleteSize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ordered_map_peek(const tensor& key, const tensor& indices, const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OrderedMapPeek", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, key.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ordered_map_size(const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OrderedMapSize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ordered_map_unstage(const tensor& key, const tensor& indices, const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OrderedMapUnstage", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, key.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor outfeed_dequeue(datatype dtype, const std::vector<int64_t>& shape, int64_t device_ordinal=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OutfeedDequeue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "device_ordinal", device_ordinal);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor outfeed_dequeue_tuple(const std::vector<datatype>& dtypes, const std::vector< std::vector<int64_t>>& shapes, int64_t device_ordinal=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "OutfeedDequeueTuple", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "device_ordinal", device_ordinal);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor pack(const std::vector<tensor>&values, int64_t axis=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Pack", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> values_handles; values_handles.reserve(values.size());
    std::transform(values.begin(), values.end(), std::back_inserter(values_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, values_handles.data(), (int)values.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", values.size());
    TFE_OpSetAttrInt(op, "axis", axis);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor pad(const tensor& input, const tensor& paddings, datatype Tpaddings=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Pad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, paddings.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tpaddings", Tpaddings);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor pad_v2(const tensor& input, const tensor& paddings, const tensor& constant_values, datatype Tpaddings=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PadV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, paddings.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, constant_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tpaddings", Tpaddings);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor padded_batch_dataset(const tensor& input_dataset, const tensor& batch_size, const std::vector<tensor>&padded_shapes, const std::vector<tensor>&padding_values, const std::vector<datatype>& Toutput_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PaddedBatchDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, batch_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> padded_shapes_handles; padded_shapes_handles.reserve((int)padded_shapes.size());
    std::transform(padded_shapes.begin(), padded_shapes.end(), std::back_inserter(padded_shapes_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, padded_shapes_handles.data(), (int)padded_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> padding_values_handles; padding_values_handles.reserve((int)padding_values.size());
    std::transform(padding_values.begin(), padding_values.end(), std::back_inserter(padding_values_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, padding_values_handles.data(), (int)padding_values.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "Toutput_types", reinterpret_cast<const enum TF_DataType *>(Toutput_types.data()), (int)Toutput_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "N", (int)padded_shapes.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor padded_batch_dataset_v2(const tensor& input_dataset, const tensor& batch_size, const std::vector<tensor>&padded_shapes, const std::vector<tensor>&padding_values, const tensor& drop_remainder, const std::vector<datatype>& Toutput_types, const std::vector< std::vector<int64_t>>& output_shapes, bool parallel_copy=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PaddedBatchDatasetV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, batch_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> padded_shapes_handles; padded_shapes_handles.reserve((int)padded_shapes.size());
    std::transform(padded_shapes.begin(), padded_shapes.end(), std::back_inserter(padded_shapes_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, padded_shapes_handles.data(), (int)padded_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> padding_values_handles; padding_values_handles.reserve((int)padding_values.size());
    std::transform(padding_values.begin(), padding_values.end(), std::back_inserter(padding_values_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, padding_values_handles.data(), (int)padding_values.size(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, drop_remainder.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "Toutput_types", reinterpret_cast<const enum TF_DataType *>(Toutput_types.data()), (int)Toutput_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "N", (int)padded_shapes.size());
    TFE_OpSetAttrBool(op, "parallel_copy", (unsigned char)parallel_copy);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor padding_f_i_f_o_queue(const std::vector<datatype>& component_types, const std::vector< std::vector<int64_t>>& shapes, int64_t capacity=-1, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PaddingFIFOQueue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor padding_f_i_f_o_queue_v2(const std::vector<datatype>& component_types, const std::vector< std::vector<int64_t>>& shapes, int64_t capacity=-1, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PaddingFIFOQueueV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor parallel_concat(const std::vector<tensor>&values, const std::vector<int64_t>& shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ParallelConcat", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> values_handles; values_handles.reserve(values.size());
    std::transform(values.begin(), values.end(), std::back_inserter(values_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, values_handles.data(), (int)values.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", values.size());
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor parallel_dynamic_stitch(const std::vector<tensor>&indices, const std::vector<tensor>&data) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ParallelDynamicStitch", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> indices_handles; indices_handles.reserve((int)indices.size());
    std::transform(indices.begin(), indices.end(), std::back_inserter(indices_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, indices_handles.data(), (int)indices.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> data_handles; data_handles.reserve((int)data.size());
    std::transform(data.begin(), data.end(), std::back_inserter(data_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, data_handles.data(), (int)data.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", indices.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor parameterized_truncated_normal(const tensor& shape, const tensor& means, const tensor& stdevs, const tensor& minvals, const tensor& maxvals, datatype dtype, int64_t seed=0, int64_t seed2=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ParameterizedTruncatedNormal", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, means.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, stdevs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, minvals.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, maxvals.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor parse_example_dataset(const tensor& input_dataset, const tensor& num_parallel_calls, const std::vector<tensor>&dense_defaults, const std::vector< std::string>& sparse_keys, const std::vector< std::string>& dense_keys, const std::vector<datatype>& sparse_types, const std::vector<datatype>& Tdense, const std::vector< std::vector<int64_t>>& dense_shapes, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, const std::vector< std::string>& ragged_keys, const std::vector<datatype>& ragged_value_types, const std::vector<datatype>& ragged_split_types, bool sloppy=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ParseExampleDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_parallel_calls.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> dense_defaults_handles; dense_defaults_handles.reserve(dense_defaults.size());
    std::transform(dense_defaults.begin(), dense_defaults.end(), std::back_inserter(dense_defaults_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, dense_defaults_handles.data(), (int)dense_defaults.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> sparse_keys_sizes; sparse_keys_sizes.reserve((int)sparse_keys.size());
    std::transform(sparse_keys.begin(), sparse_keys.end(), std::back_inserter(sparse_keys_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "sparse_keys", reinterpret_cast<const void *const *>(sparse_keys.data()), sparse_keys_sizes.data(), (int)sparse_keys.size());
    
    
    std::vector<std::size_t> dense_keys_sizes; dense_keys_sizes.reserve((int)dense_keys.size());
    std::transform(dense_keys.begin(), dense_keys.end(), std::back_inserter(dense_keys_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "dense_keys", reinterpret_cast<const void *const *>(dense_keys.data()), dense_keys_sizes.data(), (int)dense_keys.size());
    
    TFE_OpSetAttrTypeList(op, "sparse_types", reinterpret_cast<const enum TF_DataType *>(sparse_types.data()), (int)sparse_types.size());
    TFE_OpSetAttrTypeList(op, "Tdense", reinterpret_cast<const enum TF_DataType *>(Tdense.data()), (int)Tdense.size());
    
    std::vector<const int64_t*> dense_shapes_values; dense_shapes_values.reserve((int)dense_shapes.size());
    std::vector<int> dense_shapes_ndims; dense_shapes_ndims.reserve((int)dense_shapes.size());
    std::transform(dense_shapes.begin(), dense_shapes.end(), std::back_inserter(dense_shapes_values), [](const auto& v) { return v.data();});
    std::transform(dense_shapes.begin(), dense_shapes.end(), std::back_inserter(dense_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "dense_shapes", dense_shapes_values.data(), dense_shapes_ndims.data(), (int)dense_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<std::size_t> ragged_keys_sizes; ragged_keys_sizes.reserve(ragged_keys.size());
    std::transform(ragged_keys.begin(), ragged_keys.end(), std::back_inserter(ragged_keys_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "ragged_keys", reinterpret_cast<const void *const *>(ragged_keys.data()), ragged_keys_sizes.data(), (int)ragged_keys.size());
    
    TFE_OpSetAttrTypeList(op, "ragged_value_types", reinterpret_cast<const enum TF_DataType *>(ragged_value_types.data()), (int)ragged_value_types.size());
    TFE_OpSetAttrTypeList(op, "ragged_split_types", reinterpret_cast<const enum TF_DataType *>(ragged_split_types.data()), (int)ragged_split_types.size());
    TFE_OpSetAttrBool(op, "sloppy", (unsigned char)sloppy);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor parse_example_dataset_v2(const tensor& input_dataset, const tensor& num_parallel_calls, const std::vector<tensor>&dense_defaults, const std::vector< std::string>& sparse_keys, const std::vector< std::string>& dense_keys, const std::vector<datatype>& sparse_types, const std::vector<datatype>& Tdense, const std::vector< std::vector<int64_t>>& dense_shapes, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, const std::vector< std::string>& ragged_keys, const std::vector<datatype>& ragged_value_types, const std::vector<datatype>& ragged_split_types, const std::string& deterministic="default") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ParseExampleDatasetV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_parallel_calls.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> dense_defaults_handles; dense_defaults_handles.reserve((int)dense_defaults.size());
    std::transform(dense_defaults.begin(), dense_defaults.end(), std::back_inserter(dense_defaults_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, dense_defaults_handles.data(), (int)dense_defaults.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> sparse_keys_sizes; sparse_keys_sizes.reserve(sparse_keys.size());
    std::transform(sparse_keys.begin(), sparse_keys.end(), std::back_inserter(sparse_keys_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "sparse_keys", reinterpret_cast<const void *const *>(sparse_keys.data()), sparse_keys_sizes.data(), (int)sparse_keys.size());
    
    
    std::vector<std::size_t> dense_keys_sizes; dense_keys_sizes.reserve(dense_keys.size());
    std::transform(dense_keys.begin(), dense_keys.end(), std::back_inserter(dense_keys_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "dense_keys", reinterpret_cast<const void *const *>(dense_keys.data()), dense_keys_sizes.data(), (int)dense_keys.size());
    
    TFE_OpSetAttrTypeList(op, "sparse_types", reinterpret_cast<const enum TF_DataType *>(sparse_types.data()), (int)sparse_types.size());
    TFE_OpSetAttrTypeList(op, "Tdense", reinterpret_cast<const enum TF_DataType *>(Tdense.data()), (int)Tdense.size());
    
    std::vector<const int64_t*> dense_shapes_values; dense_shapes_values.reserve((int)dense_shapes.size());
    std::vector<int> dense_shapes_ndims; dense_shapes_ndims.reserve((int)dense_shapes.size());
    std::transform(dense_shapes.begin(), dense_shapes.end(), std::back_inserter(dense_shapes_values), [](const auto& v) { return v.data();});
    std::transform(dense_shapes.begin(), dense_shapes.end(), std::back_inserter(dense_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "dense_shapes", dense_shapes_values.data(), dense_shapes_ndims.data(), (int)dense_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<std::size_t> ragged_keys_sizes; ragged_keys_sizes.reserve(ragged_keys.size());
    std::transform(ragged_keys.begin(), ragged_keys.end(), std::back_inserter(ragged_keys_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "ragged_keys", reinterpret_cast<const void *const *>(ragged_keys.data()), ragged_keys_sizes.data(), (int)ragged_keys.size());
    
    TFE_OpSetAttrTypeList(op, "ragged_value_types", reinterpret_cast<const enum TF_DataType *>(ragged_value_types.data()), (int)ragged_value_types.size());
    TFE_OpSetAttrTypeList(op, "ragged_split_types", reinterpret_cast<const enum TF_DataType *>(ragged_split_types.data()), (int)ragged_split_types.size());
    TFE_OpSetAttrString(op, "deterministic", (void*) deterministic.c_str(), deterministic.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor parse_tensor(const tensor& serialized, datatype out_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ParseTensor", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, serialized.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor placeholder(datatype dtype, const std::vector<int64_t>& shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Placeholder", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor placeholder_v2(datatype dtype, const std::vector<int64_t>& shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PlaceholderV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor placeholder_with_default(const tensor& input, datatype dtype, const std::vector<int64_t>& shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PlaceholderWithDefault", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor polygamma(const tensor& a, const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Polygamma", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor population_count(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PopulationCount", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor pow(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Pow", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor prefetch_dataset(const tensor& input_dataset, const tensor& buffer_size, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, int64_t slack_period=0, bool legacy_autotune=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PrefetchDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "slack_period", slack_period);
    TFE_OpSetAttrBool(op, "legacy_autotune", (unsigned char)legacy_autotune);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor prelinearize(const tensor& input, datatype dtype, const std::vector<int64_t>& shape, const std::vector<int64_t>& layout) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Prelinearize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrIntList(op, "layout", layout.data(), (int)layout.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor prelinearize_tuple(const std::vector<tensor>&inputs, const std::vector<datatype>& dtypes, const std::vector< std::vector<int64_t>>& shapes, const std::vector<int64_t>& layouts) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PrelinearizeTuple", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> inputs_handles; inputs_handles.reserve((int)inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, inputs_handles.data(), (int)inputs.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrIntList(op, "layouts", layouts.data(), (int)layouts.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor prevent_gradient(const tensor& input, const std::string& message="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PreventGradient", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "message", (void*) message.c_str(), message.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor print(const tensor& input, const std::vector<tensor>&data, const std::vector<datatype>& U, const std::string& message="", int64_t first_n=-1, int64_t summarize=3) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Print", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> data_handles; data_handles.reserve((int)data.size());
    std::transform(data.begin(), data.end(), std::back_inserter(data_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, data_handles.data(), (int)data.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "U", reinterpret_cast<const enum TF_DataType *>(U.data()), (int)U.size());
    TFE_OpSetAttrString(op, "message", (void*) message.c_str(), message.size());
    TFE_OpSetAttrInt(op, "first_n", first_n);
    TFE_OpSetAttrInt(op, "summarize", summarize);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor priority_queue(const std::vector<datatype>& component_types, const std::vector< std::vector<int64_t>>& shapes, int64_t capacity=-1, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PriorityQueue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor priority_queue_v2(const std::vector<datatype>& component_types, const std::vector< std::vector<int64_t>>& shapes, int64_t capacity=-1, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PriorityQueueV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor private_thread_pool_dataset(const tensor& input_dataset, const tensor& num_threads, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PrivateThreadPoolDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_threads.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor prod(const tensor& input, const tensor& reduction_indices, bool keep_dims=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Prod", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor py_func(const std::vector<tensor>&input, const std::string& token, const std::vector<datatype>& Tin, const std::vector<datatype>& Tout) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PyFunc", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> input_handles; input_handles.reserve((int)input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(input_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, input_handles.data(), (int)input.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "token", (void*) token.c_str(), token.size());
    TFE_OpSetAttrTypeList(op, "Tin", reinterpret_cast<const enum TF_DataType *>(Tin.data()), (int)Tin.size());
    TFE_OpSetAttrTypeList(op, "Tout", reinterpret_cast<const enum TF_DataType *>(Tout.data()), (int)Tout.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor py_func_stateless(const std::vector<tensor>&input, const std::string& token, const std::vector<datatype>& Tin, const std::vector<datatype>& Tout) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "PyFuncStateless", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> input_handles; input_handles.reserve((int)input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(input_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, input_handles.data(), (int)input.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "token", (void*) token.c_str(), token.size());
    TFE_OpSetAttrTypeList(op, "Tin", reinterpret_cast<const enum TF_DataType *>(Tin.data()), (int)Tin.size());
    TFE_OpSetAttrTypeList(op, "Tout", reinterpret_cast<const enum TF_DataType *>(Tout.data()), (int)Tout.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor quantize_and_dequantize(const tensor& input, bool signed_input=true, int64_t num_bits=8, bool range_given=false, float input_min=0.0000e+00, float input_max=0.0000e+00) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QuantizeAndDequantize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "signed_input", (unsigned char)signed_input);
    TFE_OpSetAttrInt(op, "num_bits", num_bits);
    TFE_OpSetAttrBool(op, "range_given", (unsigned char)range_given);
    TFE_OpSetAttrFloat(op, "input_min", input_min);
    TFE_OpSetAttrFloat(op, "input_max", input_max);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor quantize_and_dequantize_v2(const tensor& input, const tensor& input_min, const tensor& input_max, bool signed_input=true, int64_t num_bits=8, bool range_given=false, const std::string& round_mode="HALF_TO_EVEN", bool narrow_range=false, int64_t axis=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QuantizeAndDequantizeV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_min.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_max.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "signed_input", (unsigned char)signed_input);
    TFE_OpSetAttrInt(op, "num_bits", num_bits);
    TFE_OpSetAttrBool(op, "range_given", (unsigned char)range_given);
    TFE_OpSetAttrString(op, "round_mode", (void*) round_mode.c_str(), round_mode.size());
    TFE_OpSetAttrBool(op, "narrow_range", (unsigned char)narrow_range);
    TFE_OpSetAttrInt(op, "axis", axis);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor quantize_and_dequantize_v3(const tensor& input, const tensor& input_min, const tensor& input_max, const tensor& num_bits, bool signed_input=true, bool range_given=true, bool narrow_range=false, int64_t axis=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QuantizeAndDequantizeV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_min.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_max.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_bits.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "signed_input", (unsigned char)signed_input);
    TFE_OpSetAttrBool(op, "range_given", (unsigned char)range_given);
    TFE_OpSetAttrBool(op, "narrow_range", (unsigned char)narrow_range);
    TFE_OpSetAttrInt(op, "axis", axis);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor quantized_mat_mul_with_bias_and_dequantize(const tensor& a, const tensor& b, const tensor& bias, const tensor& min_a, const tensor& max_a, const tensor& min_b, const tensor& max_b, const tensor& min_freezed_output, const tensor& max_freezed_output, datatype T1, datatype T2, datatype Tbias, datatype Toutput, bool transpose_a=false, bool transpose_b=false, const std::string& input_quant_mode="MIN_FIRST") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QuantizedMatMulWithBiasAndDequantize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, bias.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, min_a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, min_b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, min_freezed_output.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, max_freezed_output.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "T1", T1);
    TFE_OpSetAttrType(op, "T2", T2);
    TFE_OpSetAttrType(op, "Tbias", Tbias);
    TFE_OpSetAttrType(op, "Toutput", Toutput);
    TFE_OpSetAttrBool(op, "transpose_a", (unsigned char)transpose_a);
    TFE_OpSetAttrBool(op, "transpose_b", (unsigned char)transpose_b);
    TFE_OpSetAttrString(op, "input_quant_mode", (void*) input_quant_mode.c_str(), input_quant_mode.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor queue_dequeue(const tensor& handle, const std::vector<datatype>& component_types, int64_t timeout_ms=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QueueDequeue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    TFE_OpSetAttrInt(op, "timeout_ms", timeout_ms);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor queue_dequeue_many(const tensor& handle, const tensor& n, const std::vector<datatype>& component_types, int64_t timeout_ms=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QueueDequeueMany", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, n.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    TFE_OpSetAttrInt(op, "timeout_ms", timeout_ms);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor queue_dequeue_many_v2(const tensor& handle, const tensor& n, const std::vector<datatype>& component_types, int64_t timeout_ms=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QueueDequeueManyV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, n.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    TFE_OpSetAttrInt(op, "timeout_ms", timeout_ms);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor queue_dequeue_up_to(const tensor& handle, const tensor& n, const std::vector<datatype>& component_types, int64_t timeout_ms=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QueueDequeueUpTo", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, n.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    TFE_OpSetAttrInt(op, "timeout_ms", timeout_ms);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor queue_dequeue_up_to_v2(const tensor& handle, const tensor& n, const std::vector<datatype>& component_types, int64_t timeout_ms=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QueueDequeueUpToV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, n.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    TFE_OpSetAttrInt(op, "timeout_ms", timeout_ms);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor queue_dequeue_v2(const tensor& handle, const std::vector<datatype>& component_types, int64_t timeout_ms=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QueueDequeueV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    TFE_OpSetAttrInt(op, "timeout_ms", timeout_ms);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor queue_is_closed(const tensor& handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QueueIsClosed", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor queue_is_closed_v2(const tensor& handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QueueIsClosedV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor queue_size(const tensor& handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QueueSize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor queue_size_v2(const tensor& handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "QueueSizeV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor r_f_f_t(const tensor& input, const tensor& fft_length, datatype Treal=static_cast<datatype>(1), datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RFFT", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, fft_length.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Treal", Treal);
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor r_f_f_t2_d(const tensor& input, const tensor& fft_length, datatype Treal=static_cast<datatype>(1), datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RFFT2D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, fft_length.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Treal", Treal);
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor r_f_f_t3_d(const tensor& input, const tensor& fft_length, datatype Treal=static_cast<datatype>(1), datatype Tcomplex=static_cast<datatype>(8)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RFFT3D", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, fft_length.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Treal", Treal);
    TFE_OpSetAttrType(op, "Tcomplex", Tcomplex);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor r_g_b_to_h_s_v(const tensor& images) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RGBToHSV", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ragged_tensor_to_tensor(const tensor& shape, const tensor& values, const tensor& default_value, const std::vector<tensor>&row_partition_tensors, datatype Tindex, datatype Tshape, const std::vector< std::string>& row_partition_types) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RaggedTensorToTensor", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, default_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> row_partition_tensors_handles; row_partition_tensors_handles.reserve((int)row_partition_tensors.size());
    std::transform(row_partition_tensors.begin(), row_partition_tensors.end(), std::back_inserter(row_partition_tensors_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, row_partition_tensors_handles.data(), (int)row_partition_tensors.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindex", Tindex);
    TFE_OpSetAttrType(op, "Tshape", Tshape);
    TFE_OpSetAttrInt(op, "num_row_partition_tensors", (int)row_partition_tensors.size());
    
    std::vector<std::size_t> row_partition_types_sizes; row_partition_types_sizes.reserve((int)row_partition_types.size());
    std::transform(row_partition_types.begin(), row_partition_types.end(), std::back_inserter(row_partition_types_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "row_partition_types", reinterpret_cast<const void *const *>(row_partition_types.data()), row_partition_types_sizes.data(), (int)row_partition_types.size());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ragged_tensor_to_variant(const std::vector<tensor>&rt_nested_splits, const tensor& rt_dense_values, datatype Tvalues, bool batched_input, datatype Tsplits=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RaggedTensorToVariant", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> rt_nested_splits_handles; rt_nested_splits_handles.reserve((int)rt_nested_splits.size());
    std::transform(rt_nested_splits.begin(), rt_nested_splits.end(), std::back_inserter(rt_nested_splits_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, rt_nested_splits_handles.data(), (int)rt_nested_splits.size(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rt_dense_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "RAGGED_RANK", (int)rt_nested_splits.size());
    TFE_OpSetAttrType(op, "Tvalues", Tvalues);
    TFE_OpSetAttrBool(op, "batched_input", (unsigned char)batched_input);
    TFE_OpSetAttrType(op, "Tsplits", Tsplits);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_crop(const tensor& image, const tensor& size, int64_t seed=0, int64_t seed2=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomCrop", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, image.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_dataset(const tensor& seed, const tensor& seed2, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_gamma(const tensor& shape, const tensor& alpha, datatype S, int64_t seed=0, int64_t seed2=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomGamma", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, alpha.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "S", S);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_gamma_grad(const tensor& alpha, const tensor& sample) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomGammaGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, alpha.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sample.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_poisson(const tensor& shape, const tensor& rate, datatype S, datatype dtype, int64_t seed=0, int64_t seed2=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomPoisson", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rate.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "S", S);
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_poisson_v2(const tensor& shape, const tensor& rate, datatype S, int64_t seed=0, int64_t seed2=0, datatype R=static_cast<datatype>(2), datatype dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomPoissonV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rate.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "S", S);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);
    TFE_OpSetAttrType(op, "R", R);
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_shuffle(const tensor& value, int64_t seed=0, int64_t seed2=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomShuffle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_shuffle_queue(const std::vector<datatype>& component_types, const std::vector< std::vector<int64_t>>& shapes, int64_t capacity=-1, int64_t min_after_dequeue=0, int64_t seed=0, int64_t seed2=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomShuffleQueue", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "min_after_dequeue", min_after_dequeue);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_shuffle_queue_v2(const std::vector<datatype>& component_types, const std::vector< std::vector<int64_t>>& shapes, int64_t capacity=-1, int64_t min_after_dequeue=0, int64_t seed=0, int64_t seed2=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomShuffleQueueV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "component_types", reinterpret_cast<const enum TF_DataType *>(component_types.data()), (int)component_types.size());
    
    std::vector<const int64_t*> shapes_values; shapes_values.reserve((int)shapes.size());
    std::vector<int> shapes_ndims; shapes_ndims.reserve((int)shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_values), [](const auto& v) { return v.data();});
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "shapes", shapes_values.data(), shapes_ndims.data(), (int)shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "min_after_dequeue", min_after_dequeue);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_standard_normal(const tensor& shape, datatype dtype, int64_t seed=0, int64_t seed2=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomStandardNormal", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_uniform(const tensor& shape, datatype dtype, int64_t seed=0, int64_t seed2=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomUniform", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor random_uniform_int(const tensor& shape, const tensor& minval, const tensor& maxval, datatype Tout, int64_t seed=0, int64_t seed2=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RandomUniformInt", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, minval.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, maxval.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tout", Tout);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor range(const tensor& start, const tensor& limit, const tensor& delta, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Range", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, start.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, limit.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, delta.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor range_dataset(const tensor& start, const tensor& stop, const tensor& step, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RangeDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, start.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, stop.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, step.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor rank(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Rank", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor read_file(const tensor& filename) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReadFile", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, filename.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor read_variable_op(const tensor& resource, datatype dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReadVariableOp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reader_num_records_produced(const tensor& reader_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReaderNumRecordsProduced", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, reader_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reader_num_records_produced_v2(const tensor& reader_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReaderNumRecordsProducedV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, reader_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reader_num_work_units_completed(const tensor& reader_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReaderNumWorkUnitsCompleted", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, reader_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reader_num_work_units_completed_v2(const tensor& reader_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReaderNumWorkUnitsCompletedV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, reader_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reader_serialize_state(const tensor& reader_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReaderSerializeState", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, reader_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reader_serialize_state_v2(const tensor& reader_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReaderSerializeStateV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, reader_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor real(const tensor& input, datatype Tout=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Real", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tout", Tout);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor real_div(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RealDiv", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor rebatch_dataset(const tensor& input_dataset, const tensor& num_replicas, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, bool use_fallback=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RebatchDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_replicas.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrBool(op, "use_fallback", (unsigned char)use_fallback);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reciprocal(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Reciprocal", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reciprocal_grad(const tensor& y, const tensor& dy) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReciprocalGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dy.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor record_input(const std::string& file_pattern, int64_t file_random_seed=301, float file_shuffle_shift_ratio=0.0000e+00, int64_t file_buffer_size=10000, int64_t file_parallelism=16, int64_t batch_size=32, const std::string& compression_type="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RecordInput", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "file_pattern", (void*) file_pattern.c_str(), file_pattern.size());
    TFE_OpSetAttrInt(op, "file_random_seed", file_random_seed);
    TFE_OpSetAttrFloat(op, "file_shuffle_shift_ratio", file_shuffle_shift_ratio);
    TFE_OpSetAttrInt(op, "file_buffer_size", file_buffer_size);
    TFE_OpSetAttrInt(op, "file_parallelism", file_parallelism);
    TFE_OpSetAttrInt(op, "batch_size", batch_size);
    TFE_OpSetAttrString(op, "compression_type", (void*) compression_type.c_str(), compression_type.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor recv(datatype tensor_type, const std::string& tensor_name, const std::string& send_device, int64_t send_device_incarnation, const std::string& recv_device, bool client_terminated=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Recv", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "tensor_type", tensor_type);
    TFE_OpSetAttrString(op, "tensor_name", (void*) tensor_name.c_str(), tensor_name.size());
    TFE_OpSetAttrString(op, "send_device", (void*) send_device.c_str(), send_device.size());
    TFE_OpSetAttrInt(op, "send_device_incarnation", send_device_incarnation);
    TFE_OpSetAttrString(op, "recv_device", (void*) recv_device.c_str(), recv_device.size());
    TFE_OpSetAttrBool(op, "client_terminated", (unsigned char)client_terminated);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor recv_t_p_u_embedding_activations(int64_t num_outputs, const std::string& config) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RecvTPUEmbeddingActivations", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_outputs", num_outputs);
    TFE_OpSetAttrString(op, "config", (void*) config.c_str(), config.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reduce_join(const tensor& inputs, const tensor& reduction_indices, bool keep_dims=false, const std::string& separator="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReduceJoin", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, inputs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);
    TFE_OpSetAttrString(op, "separator", (void*) separator.c_str(), separator.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ref_enter(const tensor& data, const std::string& frame_name, bool is_constant=false, int64_t parallel_iterations=10) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RefEnter", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "frame_name", (void*) frame_name.c_str(), frame_name.size());
    TFE_OpSetAttrBool(op, "is_constant", (unsigned char)is_constant);
    TFE_OpSetAttrInt(op, "parallel_iterations", parallel_iterations);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ref_exit(const tensor& data) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RefExit", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ref_identity(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RefIdentity", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ref_next_iteration(const tensor& data) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RefNextIteration", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor ref_select(const tensor& index, const std::vector<tensor>&inputs) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RefSelect", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<TFE_TensorHandle*> inputs_handles; inputs_handles.reserve((int)inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, inputs_handles.data(), (int)inputs.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", (int)inputs.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor regex_full_match(const tensor& input, const tensor& pattern) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RegexFullMatch", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, pattern.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor regex_replace(const tensor& input, const tensor& pattern, const tensor& rewrite, bool replace_global=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RegexReplace", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, pattern.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rewrite.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "replace_global", (unsigned char)replace_global);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor relu(const tensor& features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Relu", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor relu6(const tensor& features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Relu6", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor relu6_grad(const tensor& gradients, const tensor& features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Relu6Grad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor relu_grad(const tensor& gradients, const tensor& features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReluGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor repeat_dataset(const tensor& input_dataset, const tensor& count, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RepeatDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, count.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reshape(const tensor& input_tensor, const tensor& shape, datatype Tshape=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Reshape", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tshape", Tshape);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resize_area(const tensor& images, const tensor& size, bool align_corners=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResizeArea", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "align_corners", (unsigned char)align_corners);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resize_bicubic(const tensor& images, const tensor& size, bool align_corners=false, bool half_pixel_centers=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResizeBicubic", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "align_corners", (unsigned char)align_corners);
    TFE_OpSetAttrBool(op, "half_pixel_centers", (unsigned char)half_pixel_centers);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resize_bicubic_grad(const tensor& grads, const tensor& original_image, bool align_corners=false, bool half_pixel_centers=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResizeBicubicGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, grads.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, original_image.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "align_corners", (unsigned char)align_corners);
    TFE_OpSetAttrBool(op, "half_pixel_centers", (unsigned char)half_pixel_centers);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resize_bilinear(const tensor& images, const tensor& size, bool align_corners=false, bool half_pixel_centers=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResizeBilinear", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "align_corners", (unsigned char)align_corners);
    TFE_OpSetAttrBool(op, "half_pixel_centers", (unsigned char)half_pixel_centers);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resize_bilinear_grad(const tensor& grads, const tensor& original_image, bool align_corners=false, bool half_pixel_centers=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResizeBilinearGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, grads.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, original_image.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "align_corners", (unsigned char)align_corners);
    TFE_OpSetAttrBool(op, "half_pixel_centers", (unsigned char)half_pixel_centers);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resize_nearest_neighbor(const tensor& images, const tensor& size, bool align_corners=false, bool half_pixel_centers=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResizeNearestNeighbor", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "align_corners", (unsigned char)align_corners);
    TFE_OpSetAttrBool(op, "half_pixel_centers", (unsigned char)half_pixel_centers);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resize_nearest_neighbor_grad(const tensor& grads, const tensor& size, bool align_corners=false, bool half_pixel_centers=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResizeNearestNeighborGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, grads.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "align_corners", (unsigned char)align_corners);
    TFE_OpSetAttrBool(op, "half_pixel_centers", (unsigned char)half_pixel_centers);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resource_accumulator_num_accumulated(const tensor& handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResourceAccumulatorNumAccumulated", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resource_accumulator_take_gradient(const tensor& handle, const tensor& num_required, datatype dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResourceAccumulatorTakeGradient", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_required.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resource_conditional_accumulator(datatype dtype, const std::vector<int64_t>& shape, const std::string& container="", const std::string& shared_name="", const std::string& reduction_type="MEAN") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResourceConditionalAccumulator", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrString(op, "reduction_type", (void*) reduction_type.c_str(), reduction_type.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resource_count_up_to(const tensor& resource, int64_t limit) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResourceCountUpTo", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "limit", limit);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resource_gather(const tensor& resource, const tensor& indices, datatype dtype, datatype Tindices, int64_t batch_dims=0, bool validate_indices=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResourceGather", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrInt(op, "batch_dims", batch_dims);
    TFE_OpSetAttrBool(op, "validate_indices", (unsigned char)validate_indices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor resource_gather_nd(const tensor& resource, const tensor& indices, datatype dtype, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ResourceGatherNd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor restore(const tensor& file_pattern, const tensor& input_tensor_name, datatype dt, int64_t preferred_shard=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Restore", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, file_pattern.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_tensor_name.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dt", dt);
    TFE_OpSetAttrInt(op, "preferred_shard", preferred_shard);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor restore_slice(const tensor& file_pattern, const tensor& input_tensor_name, const tensor& shape_and_slice, datatype dt, int64_t preferred_shard=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RestoreSlice", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, file_pattern.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_tensor_name.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape_and_slice.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dt", dt);
    TFE_OpSetAttrInt(op, "preferred_shard", preferred_shard);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor restore_v2(const tensor& prefix, const tensor& input_tensor_names, const tensor& shape_and_slices, const std::vector<datatype>& dtypes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RestoreV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, prefix.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_tensor_names.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape_and_slices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor retrieve_t_p_u_embedding_stochastic_gradient_descent_parameters(int64_t num_shards, int64_t shard_id, int64_t table_id=-1, const std::string& table_name="", const std::string& config="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RetrieveTPUEmbeddingStochasticGradientDescentParameters", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_shards", num_shards);
    TFE_OpSetAttrInt(op, "shard_id", shard_id);
    TFE_OpSetAttrInt(op, "table_id", table_id);
    TFE_OpSetAttrString(op, "table_name", (void*) table_name.c_str(), table_name.size());
    TFE_OpSetAttrString(op, "config", (void*) config.c_str(), config.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reverse(const tensor& input_tensor, const tensor& dims) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Reverse", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dims.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reverse_sequence(const tensor& input, const tensor& seq_lengths, int64_t seq_dim, int64_t batch_dim=0, datatype Tlen=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReverseSequence", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seq_lengths.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "seq_dim", seq_dim);
    TFE_OpSetAttrInt(op, "batch_dim", batch_dim);
    TFE_OpSetAttrType(op, "Tlen", Tlen);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor reverse_v2(const tensor& input_tensor, const tensor& axis, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ReverseV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, axis.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor right_shift(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RightShift", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor rint(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Rint", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor roll(const tensor& input, const tensor& shift, const tensor& axis, datatype Tshift, datatype Taxis) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Roll", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shift.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, axis.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tshift", Tshift);
    TFE_OpSetAttrType(op, "Taxis", Taxis);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor round(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Round", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor rsqrt(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Rsqrt", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor rsqrt_grad(const tensor& y, const tensor& dy) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "RsqrtGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dy.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sampling_dataset(const tensor& input_dataset, const tensor& rate, const tensor& seed, const tensor& seed2, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SamplingDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rate.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scalar_summary(const tensor& tags, const tensor& values) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScalarSummary", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, tags.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scale_and_translate(const tensor& images, const tensor& size, const tensor& scale, const tensor& translation, const std::string& kernel_type="lanczos3", bool antialias=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScaleAndTranslate", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, images.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, scale.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, translation.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "kernel_type", (void*) kernel_type.c_str(), kernel_type.size());
    TFE_OpSetAttrBool(op, "antialias", (unsigned char)antialias);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scale_and_translate_grad(const tensor& grads, const tensor& original_image, const tensor& scale, const tensor& translation, const std::string& kernel_type="lanczos3", bool antialias=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScaleAndTranslateGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, grads.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, original_image.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, scale.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, translation.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "kernel_type", (void*) kernel_type.c_str(), kernel_type.size());
    TFE_OpSetAttrBool(op, "antialias", (unsigned char)antialias);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_add(const tensor& ref, const tensor& indices, const tensor& updates, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterAdd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_div(const tensor& ref, const tensor& indices, const tensor& updates, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterDiv", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_max(const tensor& ref, const tensor& indices, const tensor& updates, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterMax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_min(const tensor& ref, const tensor& indices, const tensor& updates, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterMin", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_mul(const tensor& ref, const tensor& indices, const tensor& updates, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterMul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_nd(const tensor& indices, const tensor& updates, const tensor& shape, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterNd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_nd_add(const tensor& ref, const tensor& indices, const tensor& updates, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterNdAdd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_nd_non_aliasing_add(const tensor& input, const tensor& indices, const tensor& updates, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterNdNonAliasingAdd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_nd_sub(const tensor& ref, const tensor& indices, const tensor& updates, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterNdSub", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_nd_update(const tensor& ref, const tensor& indices, const tensor& updates, datatype Tindices, bool use_locking=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterNdUpdate", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_sub(const tensor& ref, const tensor& indices, const tensor& updates, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterSub", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor scatter_update(const tensor& ref, const tensor& indices, const tensor& updates, datatype Tindices, bool use_locking=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ScatterUpdate", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sdca_fprint(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SdcaFprint", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor segment_max(const tensor& data, const tensor& segment_ids, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SegmentMax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor segment_mean(const tensor& data, const tensor& segment_ids, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SegmentMean", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor segment_min(const tensor& data, const tensor& segment_ids, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SegmentMin", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor segment_prod(const tensor& data, const tensor& segment_ids, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SegmentProd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor segment_sum(const tensor& data, const tensor& segment_ids, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SegmentSum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor select(const tensor& condition, const tensor& t, const tensor& e) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Select", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, condition.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, t.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, e.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor select_v2(const tensor& condition, const tensor& t, const tensor& e) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SelectV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, condition.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, t.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, e.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor self_adjoint_eig(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SelfAdjointEig", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor selu(const tensor& features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Selu", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor selu_grad(const tensor& gradients, const tensor& outputs) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SeluGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, outputs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor serialize_iterator(const tensor& resource_handle, int64_t external_state_policy=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SerializeIterator", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "external_state_policy", external_state_policy);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor serialize_many_sparse(const tensor& sparse_indices, const tensor& sparse_values, const tensor& sparse_shape, datatype out_type=static_cast<datatype>(7)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SerializeManySparse", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sparse_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sparse_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sparse_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor serialize_sparse(const tensor& sparse_indices, const tensor& sparse_values, const tensor& sparse_shape, datatype out_type=static_cast<datatype>(7)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SerializeSparse", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sparse_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sparse_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sparse_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor serialize_tensor(const tensor& input_tensor) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SerializeTensor", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor set_size(const tensor& set_indices, const tensor& set_values, const tensor& set_shape, bool validate_indices=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SetSize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, set_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, set_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, set_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "validate_indices", (unsigned char)validate_indices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor set_stats_aggregator_dataset(const tensor& input_dataset, const tensor& stats_aggregator, const tensor& tag, const tensor& counter_prefix, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SetStatsAggregatorDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, stats_aggregator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, counter_prefix.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor shape(const tensor& input, datatype out_type=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Shape", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor shape_n(const std::vector<tensor>&input, datatype out_type=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ShapeN", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> input_handles; input_handles.reserve((int)input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(input_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, input_handles.data(), (int)input.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", (int)input.size());
    TFE_OpSetAttrType(op, "out_type", out_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor shard_dataset(const tensor& input_dataset, const tensor& num_shards, const tensor& index, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, bool require_non_empty=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ShardDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_shards.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrBool(op, "require_non_empty", (unsigned char)require_non_empty);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sharded_filename(const tensor& basename, const tensor& shard, const tensor& num_shards) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ShardedFilename", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, basename.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shard.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_shards.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sharded_filespec(const tensor& basename, const tensor& num_shards) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ShardedFilespec", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, basename.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_shards.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor shuffle_and_repeat_dataset(const tensor& input_dataset, const tensor& buffer_size, const tensor& seed, const tensor& seed2, const tensor& count, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, bool reshuffle_each_iteration=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ShuffleAndRepeatDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, count.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrBool(op, "reshuffle_each_iteration", (unsigned char)reshuffle_each_iteration);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor shuffle_dataset(const tensor& input_dataset, const tensor& buffer_size, const tensor& seed, const tensor& seed2, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, bool reshuffle_each_iteration=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ShuffleDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrBool(op, "reshuffle_each_iteration", (unsigned char)reshuffle_each_iteration);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor shuffle_dataset_v2(const tensor& input_dataset, const tensor& buffer_size, const tensor& seed_generator, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ShuffleDatasetV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed_generator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sigmoid(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Sigmoid", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sigmoid_grad(const tensor& y, const tensor& dy) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SigmoidGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dy.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sign(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Sign", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sin(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Sin", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sinh(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Sinh", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor size(const tensor& input, datatype out_type=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Size", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor skip_dataset(const tensor& input_dataset, const tensor& count, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SkipDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, count.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sleep_dataset(const tensor& input_dataset, const tensor& sleep_microseconds, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SleepDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sleep_microseconds.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor slice(const tensor& input, const tensor& begin, const tensor& size, datatype Index) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Slice", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, begin.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Index", Index);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sliding_window_dataset(const tensor& input_dataset, const tensor& window_size, const tensor& window_shift, const tensor& window_stride, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SlidingWindowDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, window_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, window_shift.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, window_stride.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor snapshot(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Snapshot", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor snapshot_dataset(const tensor& input_dataset, const tensor& path, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes, const std::string& compression="", const std::string& reader_path_prefix="", const std::string& writer_path_prefix="", int64_t shard_size_bytes=10737418240, int64_t pending_snapshot_expiry_seconds=86400, int64_t num_reader_threads=1, int64_t reader_buffer_size=1, int64_t num_writer_threads=1, int64_t writer_buffer_size=1, bool shuffle_on_read=false, int64_t seed=0, int64_t seed2=0, const std::string& mode="auto", const std::string& snapshot_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SnapshotDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, path.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "compression", (void*) compression.c_str(), compression.size());
    TFE_OpSetAttrString(op, "reader_path_prefix", (void*) reader_path_prefix.c_str(), reader_path_prefix.size());
    TFE_OpSetAttrString(op, "writer_path_prefix", (void*) writer_path_prefix.c_str(), writer_path_prefix.size());
    TFE_OpSetAttrInt(op, "shard_size_bytes", shard_size_bytes);
    TFE_OpSetAttrInt(op, "pending_snapshot_expiry_seconds", pending_snapshot_expiry_seconds);
    TFE_OpSetAttrInt(op, "num_reader_threads", num_reader_threads);
    TFE_OpSetAttrInt(op, "reader_buffer_size", reader_buffer_size);
    TFE_OpSetAttrInt(op, "num_writer_threads", num_writer_threads);
    TFE_OpSetAttrInt(op, "writer_buffer_size", writer_buffer_size);
    TFE_OpSetAttrBool(op, "shuffle_on_read", (unsigned char)shuffle_on_read);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);
    TFE_OpSetAttrString(op, "mode", (void*) mode.c_str(), mode.size());
    TFE_OpSetAttrString(op, "snapshot_name", (void*) snapshot_name.c_str(), snapshot_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sobol_sample(const tensor& dim, const tensor& num_results, const tensor& skip, datatype dtype=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SobolSample", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, dim.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_results.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, skip.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor softmax(const tensor& logits) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Softmax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, logits.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor softplus(const tensor& features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Softplus", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor softplus_grad(const tensor& gradients, const tensor& features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SoftplusGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor softsign(const tensor& features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Softsign", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor softsign_grad(const tensor& gradients, const tensor& features) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SoftsignGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, gradients.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, features.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor space_to_batch(const tensor& input, const tensor& paddings, int64_t block_size, datatype Tpaddings=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SpaceToBatch", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, paddings.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "block_size", block_size);
    TFE_OpSetAttrType(op, "Tpaddings", Tpaddings);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor space_to_batch_n_d(const tensor& input, const tensor& block_shape, const tensor& paddings, datatype Tblock_shape=static_cast<datatype>(3), datatype Tpaddings=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SpaceToBatchND", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, block_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, paddings.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tblock_shape", Tblock_shape);
    TFE_OpSetAttrType(op, "Tpaddings", Tpaddings);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor space_to_depth(const tensor& input, int64_t block_size, const std::string& data_format="NHWC") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SpaceToDepth", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "block_size", block_size);
    TFE_OpSetAttrString(op, "data_format", (void*) data_format.c_str(), data_format.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_adadelta(const tensor& var, const tensor& accum, const tensor& accum_update, const tensor& lr, const tensor& rho, const tensor& epsilon, const tensor& grad, const tensor& indices, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyAdadelta", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum_update.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rho.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_adagrad(const tensor& var, const tensor& accum, const tensor& lr, const tensor& grad, const tensor& indices, datatype Tindices, bool use_locking=false, bool update_slots=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyAdagrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "update_slots", (unsigned char)update_slots);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_adagrad_d_a(const tensor& var, const tensor& gradient_accumulator, const tensor& gradient_squared_accumulator, const tensor& grad, const tensor& indices, const tensor& lr, const tensor& l1, const tensor& l2, const tensor& global_step, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyAdagradDA", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, gradient_accumulator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, gradient_squared_accumulator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, global_step.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_adagrad_v2(const tensor& var, const tensor& accum, const tensor& lr, const tensor& epsilon, const tensor& grad, const tensor& indices, datatype Tindices, bool use_locking=false, bool update_slots=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyAdagradV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "update_slots", (unsigned char)update_slots);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_centered_r_m_s_prop(const tensor& var, const tensor& mg, const tensor& ms, const tensor& mom, const tensor& lr, const tensor& rho, const tensor& momentum, const tensor& epsilon, const tensor& grad, const tensor& indices, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyCenteredRMSProp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, mg.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, ms.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, mom.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rho.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, momentum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_ftrl(const tensor& var, const tensor& accum, const tensor& linear, const tensor& grad, const tensor& indices, const tensor& lr, const tensor& l1, const tensor& l2, const tensor& lr_power, datatype Tindices, bool use_locking=false, bool multiply_linear_by_lr=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyFtrl", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, linear.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr_power.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "multiply_linear_by_lr", (unsigned char)multiply_linear_by_lr);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_ftrl_v2(const tensor& var, const tensor& accum, const tensor& linear, const tensor& grad, const tensor& indices, const tensor& lr, const tensor& l1, const tensor& l2, const tensor& l2_shrinkage, const tensor& lr_power, datatype Tindices, bool use_locking=false, bool multiply_linear_by_lr=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyFtrlV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, linear.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2_shrinkage.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr_power.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "multiply_linear_by_lr", (unsigned char)multiply_linear_by_lr);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_momentum(const tensor& var, const tensor& accum, const tensor& lr, const tensor& grad, const tensor& indices, const tensor& momentum, datatype Tindices, bool use_locking=false, bool use_nesterov=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyMomentum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, momentum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);
    TFE_OpSetAttrBool(op, "use_nesterov", (unsigned char)use_nesterov);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_proximal_adagrad(const tensor& var, const tensor& accum, const tensor& lr, const tensor& l1, const tensor& l2, const tensor& grad, const tensor& indices, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyProximalAdagrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, accum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_proximal_gradient_descent(const tensor& var, const tensor& alpha, const tensor& l1, const tensor& l2, const tensor& grad, const tensor& indices, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyProximalGradientDescent", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, alpha.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l1.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, l2.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_apply_r_m_s_prop(const tensor& var, const tensor& ms, const tensor& mom, const tensor& lr, const tensor& rho, const tensor& momentum, const tensor& epsilon, const tensor& grad, const tensor& indices, datatype Tindices, bool use_locking=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseApplyRMSProp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, var.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, ms.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, mom.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lr.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rho.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, momentum.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, epsilon.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "use_locking", (unsigned char)use_locking);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_conditional_accumulator(datatype dtype, const std::vector<int64_t>& shape, const std::string& container="", const std::string& shared_name="", const std::string& reduction_type="MEAN") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseConditionalAccumulator", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrString(op, "reduction_type", (void*) reduction_type.c_str(), reduction_type.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_dense_cwise_add(const tensor& sp_indices, const tensor& sp_values, const tensor& sp_shape, const tensor& dense) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseDenseCwiseAdd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sp_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sp_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sp_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dense.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_dense_cwise_div(const tensor& sp_indices, const tensor& sp_values, const tensor& sp_shape, const tensor& dense) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseDenseCwiseDiv", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sp_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sp_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sp_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dense.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_dense_cwise_mul(const tensor& sp_indices, const tensor& sp_values, const tensor& sp_shape, const tensor& dense) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseDenseCwiseMul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sp_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sp_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sp_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dense.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_mat_mul(const tensor& a, const tensor& b, bool transpose_a=false, bool transpose_b=false, bool a_is_sparse=false, bool b_is_sparse=false, datatype Ta=static_cast<datatype>(1), datatype Tb=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatMul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "transpose_a", (unsigned char)transpose_a);
    TFE_OpSetAttrBool(op, "transpose_b", (unsigned char)transpose_b);
    TFE_OpSetAttrBool(op, "a_is_sparse", (unsigned char)a_is_sparse);
    TFE_OpSetAttrBool(op, "b_is_sparse", (unsigned char)b_is_sparse);
    TFE_OpSetAttrType(op, "Ta", Ta);
    TFE_OpSetAttrType(op, "Tb", Tb);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_add(const tensor& a, const tensor& b, const tensor& alpha, const tensor& beta) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixAdd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, alpha.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, beta.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_mat_mul(const tensor& a, const tensor& b, bool transpose_a=false, bool transpose_b=false, bool adjoint_a=false, bool adjoint_b=false, bool transpose_output=false, bool conjugate_output=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixMatMul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "transpose_a", (unsigned char)transpose_a);
    TFE_OpSetAttrBool(op, "transpose_b", (unsigned char)transpose_b);
    TFE_OpSetAttrBool(op, "adjoint_a", (unsigned char)adjoint_a);
    TFE_OpSetAttrBool(op, "adjoint_b", (unsigned char)adjoint_b);
    TFE_OpSetAttrBool(op, "transpose_output", (unsigned char)transpose_output);
    TFE_OpSetAttrBool(op, "conjugate_output", (unsigned char)conjugate_output);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_mul(const tensor& a, const tensor& b) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixMul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_n_n_z(const tensor& sparse_matrix) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixNNZ", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sparse_matrix.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_ordering_a_m_d(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixOrderingAMD", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_softmax(const tensor& logits, datatype type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixSoftmax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, logits.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "type", type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_softmax_grad(const tensor& softmax, const tensor& grad_softmax, datatype type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixSoftmaxGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, softmax.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad_softmax.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "type", type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_sparse_cholesky(const tensor& input, const tensor& permutation, datatype type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixSparseCholesky", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, permutation.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "type", type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_sparse_mat_mul(const tensor& a, const tensor& b, datatype type, bool transpose_a=false, bool transpose_b=false, bool adjoint_a=false, bool adjoint_b=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixSparseMatMul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "type", type);
    TFE_OpSetAttrBool(op, "transpose_a", (unsigned char)transpose_a);
    TFE_OpSetAttrBool(op, "transpose_b", (unsigned char)transpose_b);
    TFE_OpSetAttrBool(op, "adjoint_a", (unsigned char)adjoint_a);
    TFE_OpSetAttrBool(op, "adjoint_b", (unsigned char)adjoint_b);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_transpose(const tensor& input, datatype type, bool conjugate=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixTranspose", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "type", type);
    TFE_OpSetAttrBool(op, "conjugate", (unsigned char)conjugate);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_matrix_zeros(const tensor& dense_shape, datatype type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseMatrixZeros", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, dense_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "type", type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_reduce_max(const tensor& input_indices, const tensor& input_values, const tensor& input_shape, const tensor& reduction_axes, bool keep_dims=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseReduceMax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_axes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_reduce_sum(const tensor& input_indices, const tensor& input_values, const tensor& input_shape, const tensor& reduction_axes, bool keep_dims=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseReduceSum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_axes.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_segment_mean(const tensor& data, const tensor& indices, const tensor& segment_ids, datatype Tidx=static_cast<datatype>(3), datatype Tsegmentids=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseSegmentMean", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);
    TFE_OpSetAttrType(op, "Tsegmentids", Tsegmentids);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_segment_mean_grad(const tensor& grad, const tensor& indices, const tensor& segment_ids, const tensor& output_dim0, datatype Tidx=static_cast<datatype>(3), datatype Tsegmentids=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseSegmentMeanGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, output_dim0.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);
    TFE_OpSetAttrType(op, "Tsegmentids", Tsegmentids);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_segment_mean_with_num_segments(const tensor& data, const tensor& indices, const tensor& segment_ids, const tensor& num_segments, datatype Tidx=static_cast<datatype>(3), datatype Tnumsegments=static_cast<datatype>(3), datatype Tsegmentids=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseSegmentMeanWithNumSegments", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_segments.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);
    TFE_OpSetAttrType(op, "Tnumsegments", Tnumsegments);
    TFE_OpSetAttrType(op, "Tsegmentids", Tsegmentids);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_segment_sqrt_n(const tensor& data, const tensor& indices, const tensor& segment_ids, datatype Tidx=static_cast<datatype>(3), datatype Tsegmentids=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseSegmentSqrtN", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);
    TFE_OpSetAttrType(op, "Tsegmentids", Tsegmentids);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_segment_sqrt_n_grad(const tensor& grad, const tensor& indices, const tensor& segment_ids, const tensor& output_dim0, datatype Tidx=static_cast<datatype>(3), datatype Tsegmentids=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseSegmentSqrtNGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, output_dim0.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);
    TFE_OpSetAttrType(op, "Tsegmentids", Tsegmentids);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_segment_sqrt_n_with_num_segments(const tensor& data, const tensor& indices, const tensor& segment_ids, const tensor& num_segments, datatype Tidx=static_cast<datatype>(3), datatype Tnumsegments=static_cast<datatype>(3), datatype Tsegmentids=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseSegmentSqrtNWithNumSegments", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_segments.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);
    TFE_OpSetAttrType(op, "Tnumsegments", Tnumsegments);
    TFE_OpSetAttrType(op, "Tsegmentids", Tsegmentids);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_segment_sum(const tensor& data, const tensor& indices, const tensor& segment_ids, datatype Tidx=static_cast<datatype>(3), datatype Tsegmentids=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseSegmentSum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);
    TFE_OpSetAttrType(op, "Tsegmentids", Tsegmentids);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_segment_sum_with_num_segments(const tensor& data, const tensor& indices, const tensor& segment_ids, const tensor& num_segments, datatype Tidx=static_cast<datatype>(3), datatype Tnumsegments=static_cast<datatype>(3), datatype Tsegmentids=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseSegmentSumWithNumSegments", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_segments.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);
    TFE_OpSetAttrType(op, "Tnumsegments", Tnumsegments);
    TFE_OpSetAttrType(op, "Tsegmentids", Tsegmentids);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_slice_grad(const tensor& backprop_val_grad, const tensor& input_indices, const tensor& input_start, const tensor& output_indices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseSliceGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, backprop_val_grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_start.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, output_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_softmax(const tensor& sp_indices, const tensor& sp_values, const tensor& sp_shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseSoftmax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sp_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sp_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sp_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_tensor_dense_add(const tensor& a_indices, const tensor& a_values, const tensor& a_shape, const tensor& b, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseTensorDenseAdd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, a_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, a_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_tensor_dense_mat_mul(const tensor& a_indices, const tensor& a_values, const tensor& a_shape, const tensor& b, datatype Tindices=static_cast<datatype>(9), bool adjoint_a=false, bool adjoint_b=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseTensorDenseMatMul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, a_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, a_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, a_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "adjoint_a", (unsigned char)adjoint_a);
    TFE_OpSetAttrBool(op, "adjoint_b", (unsigned char)adjoint_b);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_tensor_slice_dataset(const tensor& indices, const tensor& values, const tensor& dense_shape, datatype Tvalues) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseTensorSliceDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dense_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tvalues", Tvalues);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_tensor_to_c_s_r_sparse_matrix(const tensor& indices, const tensor& values, const tensor& dense_shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseTensorToCSRSparseMatrix", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dense_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sparse_to_dense(const tensor& sparse_indices, const tensor& output_shape, const tensor& sparse_values, const tensor& default_value, datatype Tindices, bool validate_indices=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SparseToDense", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sparse_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, output_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sparse_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, default_value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrBool(op, "validate_indices", (unsigned char)validate_indices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor spence(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Spence", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor split(const tensor& split_dim, const tensor& value, int64_t num_split) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Split", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, split_dim.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_split", num_split);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor split_v(const tensor& value, const tensor& size_splits, const tensor& split_dim, int64_t num_split, datatype Tlen=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SplitV", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size_splits.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, split_dim.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_split", num_split);
    TFE_OpSetAttrType(op, "Tlen", Tlen);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sql_dataset(const tensor& driver_name, const tensor& data_source_name, const tensor& query, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SqlDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, driver_name.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, data_source_name.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, query.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sqrt(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Sqrt", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sqrt_grad(const tensor& y, const tensor& dy) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SqrtGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dy.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor square(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Square", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor squared_difference(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SquaredDifference", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor squeeze(const tensor& input, const std::vector<int64_t>& squeeze_dims) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Squeeze", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrIntList(op, "squeeze_dims", squeeze_dims.data(), (int)squeeze_dims.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stack(datatype elem_type, const std::string& stack_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Stack", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "elem_type", elem_type);
    TFE_OpSetAttrString(op, "stack_name", (void*) stack_name.c_str(), stack_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stack_pop(const tensor& handle, datatype elem_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StackPop", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "elem_type", elem_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stack_pop_v2(const tensor& handle, datatype elem_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StackPopV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "elem_type", elem_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stack_push(const tensor& handle, const tensor& elem, bool swap_memory=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StackPush", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, elem.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "swap_memory", (unsigned char)swap_memory);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stack_push_v2(const tensor& handle, const tensor& elem, bool swap_memory=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StackPushV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, elem.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "swap_memory", (unsigned char)swap_memory);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stack_v2(const tensor& max_size, datatype elem_type, const std::string& stack_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StackV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, max_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "elem_type", elem_type);
    TFE_OpSetAttrString(op, "stack_name", (void*) stack_name.c_str(), stack_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stage_peek(const tensor& index, const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StagePeek", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stage_size(const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StageSize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateful_random_binomial(const tensor& resource, const tensor& algorithm, const tensor& shape, const tensor& counts, const tensor& probs, datatype S, datatype dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatefulRandomBinomial", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, algorithm.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, counts.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, probs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "S", S);
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateful_standard_normal(const tensor& resource, const tensor& shape, datatype dtype=static_cast<datatype>(1), datatype shape_dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatefulStandardNormal", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "shape_dtype", shape_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateful_standard_normal_v2(const tensor& resource, const tensor& algorithm, const tensor& shape, datatype dtype=static_cast<datatype>(1), datatype shape_dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatefulStandardNormalV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, algorithm.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "shape_dtype", shape_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateful_truncated_normal(const tensor& resource, const tensor& algorithm, const tensor& shape, datatype dtype=static_cast<datatype>(1), datatype shape_dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatefulTruncatedNormal", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, algorithm.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "shape_dtype", shape_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateful_uniform(const tensor& resource, const tensor& algorithm, const tensor& shape, datatype dtype=static_cast<datatype>(1), datatype shape_dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatefulUniform", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, algorithm.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "shape_dtype", shape_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateful_uniform_full_int(const tensor& resource, const tensor& algorithm, const tensor& shape, datatype dtype=static_cast<datatype>(23), datatype shape_dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatefulUniformFullInt", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, algorithm.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "shape_dtype", shape_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateful_uniform_int(const tensor& resource, const tensor& algorithm, const tensor& shape, const tensor& minval, const tensor& maxval, datatype dtype=static_cast<datatype>(9), datatype shape_dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatefulUniformInt", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, algorithm.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, minval.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, maxval.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "shape_dtype", shape_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateless_multinomial(const tensor& logits, const tensor& num_samples, const tensor& seed, datatype Tseed=static_cast<datatype>(9), datatype output_dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatelessMultinomial", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, logits.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_samples.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tseed", Tseed);
    TFE_OpSetAttrType(op, "output_dtype", output_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateless_random_binomial(const tensor& shape, const tensor& seed, const tensor& counts, const tensor& probs, datatype S, datatype Tseed=static_cast<datatype>(9), datatype dtype=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatelessRandomBinomial", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, counts.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, probs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "S", S);
    TFE_OpSetAttrType(op, "Tseed", Tseed);
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateless_random_gamma_v2(const tensor& shape, const tensor& seed, const tensor& alpha, datatype dtype, datatype Tseed=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatelessRandomGammaV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, alpha.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "Tseed", Tseed);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateless_random_normal(const tensor& shape, const tensor& seed, datatype dtype=static_cast<datatype>(1), datatype Tseed=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatelessRandomNormal", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "Tseed", Tseed);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateless_random_poisson(const tensor& shape, const tensor& seed, const tensor& lam, datatype Rtype, datatype dtype, datatype Tseed=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatelessRandomPoisson", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lam.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Rtype", Rtype);
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "Tseed", Tseed);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateless_random_uniform(const tensor& shape, const tensor& seed, datatype dtype=static_cast<datatype>(1), datatype Tseed=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatelessRandomUniform", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "Tseed", Tseed);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateless_random_uniform_full_int(const tensor& shape, const tensor& seed, datatype dtype=static_cast<datatype>(23), datatype Tseed=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatelessRandomUniformFullInt", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "Tseed", Tseed);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateless_random_uniform_int(const tensor& shape, const tensor& seed, const tensor& minval, const tensor& maxval, datatype dtype, datatype Tseed=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatelessRandomUniformInt", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, minval.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, maxval.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "Tseed", Tseed);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stateless_truncated_normal(const tensor& shape, const tensor& seed, datatype dtype=static_cast<datatype>(1), datatype Tseed=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatelessTruncatedNormal", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, seed.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrType(op, "Tseed", Tseed);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor static_regex_full_match(const tensor& input, const std::string& pattern) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StaticRegexFullMatch", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "pattern", (void*) pattern.c_str(), pattern.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor static_regex_replace(const tensor& input, const std::string& pattern, const std::string& rewrite, bool replace_global=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StaticRegexReplace", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "pattern", (void*) pattern.c_str(), pattern.size());
    TFE_OpSetAttrString(op, "rewrite", (void*) rewrite.c_str(), rewrite.size());
    TFE_OpSetAttrBool(op, "replace_global", (unsigned char)replace_global);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stats_aggregator_handle(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatsAggregatorHandle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stats_aggregator_handle_v2(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatsAggregatorHandleV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stats_aggregator_summary(const tensor& iterator) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StatsAggregatorSummary", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, iterator.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor stop_gradient(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StopGradient", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor strided_slice(const tensor& input, const tensor& begin, const tensor& end, const tensor& strides, datatype Index, int64_t begin_mask=0, int64_t end_mask=0, int64_t ellipsis_mask=0, int64_t new_axis_mask=0, int64_t shrink_axis_mask=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StridedSlice", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, begin.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, end.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, strides.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Index", Index);
    TFE_OpSetAttrInt(op, "begin_mask", begin_mask);
    TFE_OpSetAttrInt(op, "end_mask", end_mask);
    TFE_OpSetAttrInt(op, "ellipsis_mask", ellipsis_mask);
    TFE_OpSetAttrInt(op, "new_axis_mask", new_axis_mask);
    TFE_OpSetAttrInt(op, "shrink_axis_mask", shrink_axis_mask);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor strided_slice_assign(const tensor& ref, const tensor& begin, const tensor& end, const tensor& strides, const tensor& value, datatype Index, int64_t begin_mask=0, int64_t end_mask=0, int64_t ellipsis_mask=0, int64_t new_axis_mask=0, int64_t shrink_axis_mask=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StridedSliceAssign", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, ref.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, begin.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, end.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, strides.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Index", Index);
    TFE_OpSetAttrInt(op, "begin_mask", begin_mask);
    TFE_OpSetAttrInt(op, "end_mask", end_mask);
    TFE_OpSetAttrInt(op, "ellipsis_mask", ellipsis_mask);
    TFE_OpSetAttrInt(op, "new_axis_mask", new_axis_mask);
    TFE_OpSetAttrInt(op, "shrink_axis_mask", shrink_axis_mask);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor strided_slice_grad(const tensor& shape, const tensor& begin, const tensor& end, const tensor& strides, const tensor& dy, datatype Index, int64_t begin_mask=0, int64_t end_mask=0, int64_t ellipsis_mask=0, int64_t new_axis_mask=0, int64_t shrink_axis_mask=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StridedSliceGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, begin.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, end.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, strides.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dy.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Index", Index);
    TFE_OpSetAttrInt(op, "begin_mask", begin_mask);
    TFE_OpSetAttrInt(op, "end_mask", end_mask);
    TFE_OpSetAttrInt(op, "ellipsis_mask", ellipsis_mask);
    TFE_OpSetAttrInt(op, "new_axis_mask", new_axis_mask);
    TFE_OpSetAttrInt(op, "shrink_axis_mask", shrink_axis_mask);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor string_format(const std::vector<tensor>&inputs, const std::string& template_arg="%s", const std::string& placeholder="%s", int64_t summarize=3) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StringFormat", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> inputs_handles; inputs_handles.reserve((int)inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, inputs_handles.data(), (int)inputs.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "template", (void*) template_arg.c_str(), template_arg.size());
    TFE_OpSetAttrString(op, "placeholder", (void*) placeholder.c_str(), placeholder.size());
    TFE_OpSetAttrInt(op, "summarize", summarize);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor string_join(const std::vector<tensor>&inputs, const std::string& separator="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StringJoin", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> inputs_handles; inputs_handles.reserve((int)inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, inputs_handles.data(), (int)inputs.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", (int)inputs.size());
    TFE_OpSetAttrString(op, "separator", (void*) separator.c_str(), separator.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor string_length(const tensor& input, const std::string& unit="BYTE") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StringLength", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "unit", (void*) unit.c_str(), unit.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor string_lower(const tensor& input, const std::string& encoding="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StringLower", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "encoding", (void*) encoding.c_str(), encoding.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor string_strip(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StringStrip", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor string_to_hash_bucket(const tensor& string_input_tensor, int64_t num_buckets) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StringToHashBucket", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, string_input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_buckets", num_buckets);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor string_to_hash_bucket_fast(const tensor& input, int64_t num_buckets) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StringToHashBucketFast", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_buckets", num_buckets);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor string_to_hash_bucket_strong(const tensor& input, int64_t num_buckets, const std::vector<int64_t>& key) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StringToHashBucketStrong", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_buckets", num_buckets);
    TFE_OpSetAttrIntList(op, "key", key.data(), (int)key.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor string_to_number(const tensor& string_input_tensor, datatype out_type=static_cast<datatype>(1)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StringToNumber", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, string_input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor string_upper(const tensor& input, const std::string& encoding="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "StringUpper", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "encoding", (void*) encoding.c_str(), encoding.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sub(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Sub", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor substr(const tensor& input, const tensor& pos, const tensor& len, const std::string& unit="BYTE") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Substr", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, pos.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, len.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "unit", (void*) unit.c_str(), unit.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor sum(const tensor& input, const tensor& reduction_indices, bool keep_dims=false, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Sum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, reduction_indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "keep_dims", (unsigned char)keep_dims);
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor summary_writer(const std::string& shared_name="", const std::string& container="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "SummaryWriter", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor t_f_record_dataset(const tensor& filenames, const tensor& compression_type, const tensor& buffer_size) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TFRecordDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, filenames.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, compression_type.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor t_f_record_reader(const std::string& container="", const std::string& shared_name="", const std::string& compression_type="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TFRecordReader", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrString(op, "compression_type", (void*) compression_type.c_str(), compression_type.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor t_f_record_reader_v2(const std::string& container="", const std::string& shared_name="", const std::string& compression_type="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TFRecordReaderV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());
    TFE_OpSetAttrString(op, "compression_type", (void*) compression_type.c_str(), compression_type.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor t_p_u_compilation_result() {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TPUCompilationResult", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor t_p_u_embedding_activations(const tensor& embedding_variable, const tensor& sliced_activations, int64_t table_id, int64_t lookup_id) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TPUEmbeddingActivations", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, embedding_variable.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, sliced_activations.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "table_id", table_id);
    TFE_OpSetAttrInt(op, "lookup_id", lookup_id);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor t_p_u_ordinal_selector() {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TPUOrdinalSelector", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor t_p_u_replicated_input(const std::vector<tensor>&inputs, bool is_mirrored_variable=false, int64_t index=-1, bool is_packed=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TPUReplicatedInput", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> inputs_handles; inputs_handles.reserve((int)inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, inputs_handles.data(), (int)inputs.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "N", (int)inputs.size());
    TFE_OpSetAttrBool(op, "is_mirrored_variable", (unsigned char)is_mirrored_variable);
    TFE_OpSetAttrInt(op, "index", index);
    TFE_OpSetAttrBool(op, "is_packed", (unsigned char)is_packed);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor t_p_u_replicated_output(const tensor& input, int64_t num_replicas) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TPUReplicatedOutput", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_replicas", num_replicas);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor take_dataset(const tensor& input_dataset, const tensor& count, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TakeDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, count.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tan(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Tan", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tanh(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Tanh", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tanh_grad(const tensor& y, const tensor& dy) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TanhGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dy.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor temporary_variable(const std::vector<int64_t>& shape, datatype dtype, const std::string& var_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TemporaryVariable", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrString(op, "var_name", (void*) var_name.c_str(), var_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array(const tensor& size, datatype dtype, const std::vector<int64_t>& element_shape, bool dynamic_size=false, bool clear_after_read=true, const std::string& tensor_array_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArray", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "element_shape", element_shape.data(), (int)element_shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrBool(op, "dynamic_size", (unsigned char)dynamic_size);
    TFE_OpSetAttrBool(op, "clear_after_read", (unsigned char)clear_after_read);
    TFE_OpSetAttrString(op, "tensor_array_name", (void*) tensor_array_name.c_str(), tensor_array_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_gather(const tensor& handle, const tensor& indices, const tensor& flow_in, datatype dtype, const std::vector<int64_t>& element_shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayGather", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "element_shape", element_shape.data(), (int)element_shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_gather_v2(const tensor& handle, const tensor& indices, const tensor& flow_in, datatype dtype, const std::vector<int64_t>& element_shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayGatherV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "element_shape", element_shape.data(), (int)element_shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_gather_v3(const tensor& handle, const tensor& indices, const tensor& flow_in, datatype dtype, const std::vector<int64_t>& element_shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayGatherV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "element_shape", element_shape.data(), (int)element_shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_grad(const tensor& handle, const tensor& flow_in, const std::string& source) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "source", (void*) source.c_str(), source.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_grad_v2(const tensor& handle, const tensor& flow_in, const std::string& source) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayGradV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "source", (void*) source.c_str(), source.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_pack(const tensor& handle, const tensor& flow_in, datatype dtype, const std::vector<int64_t>& element_shape) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayPack", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "element_shape", element_shape.data(), (int)element_shape.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_read(const tensor& handle, const tensor& index, const tensor& flow_in, datatype dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayRead", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_read_v2(const tensor& handle, const tensor& index, const tensor& flow_in, datatype dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayReadV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_read_v3(const tensor& handle, const tensor& index, const tensor& flow_in, datatype dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayReadV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_scatter(const tensor& handle, const tensor& indices, const tensor& value, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayScatter", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_scatter_v2(const tensor& handle, const tensor& indices, const tensor& value, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayScatterV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_scatter_v3(const tensor& handle, const tensor& indices, const tensor& value, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayScatterV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_size(const tensor& handle, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArraySize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_size_v2(const tensor& handle, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArraySizeV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_size_v3(const tensor& handle, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArraySizeV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_split(const tensor& handle, const tensor& value, const tensor& lengths, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArraySplit", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lengths.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_split_v2(const tensor& handle, const tensor& value, const tensor& lengths, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArraySplitV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lengths.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_split_v3(const tensor& handle, const tensor& value, const tensor& lengths, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArraySplitV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lengths.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_unpack(const tensor& handle, const tensor& value, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayUnpack", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_v2(const tensor& size, datatype dtype, const std::vector<int64_t>& element_shape, bool dynamic_size=false, bool clear_after_read=true, const std::string& tensor_array_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "element_shape", element_shape.data(), (int)element_shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrBool(op, "dynamic_size", (unsigned char)dynamic_size);
    TFE_OpSetAttrBool(op, "clear_after_read", (unsigned char)clear_after_read);
    TFE_OpSetAttrString(op, "tensor_array_name", (void*) tensor_array_name.c_str(), tensor_array_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_write(const tensor& handle, const tensor& index, const tensor& value, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayWrite", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_write_v2(const tensor& handle, const tensor& index, const tensor& value, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayWriteV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_array_write_v3(const tensor& handle, const tensor& index, const tensor& value, const tensor& flow_in) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorArrayWriteV3", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, flow_in.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_dataset(const std::vector<tensor>&components, const std::vector<datatype>& Toutput_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> components_handles; components_handles.reserve((int)components.size());
    std::transform(components.begin(), components.end(), std::back_inserter(components_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, components_handles.data(), (int)components.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "Toutput_types", reinterpret_cast<const enum TF_DataType *>(Toutput_types.data()), (int)Toutput_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_concat_lists(const tensor& input_a, const tensor& input_b, datatype element_dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListConcatLists", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_a.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_b.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_element_shape(const tensor& input_handle, datatype shape_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListElementShape", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "shape_type", shape_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_from_tensor(const tensor& input_tensor, const tensor& element_shape, datatype element_dtype, datatype shape_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListFromTensor", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, element_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);
    TFE_OpSetAttrType(op, "shape_type", shape_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_gather(const tensor& input_handle, const tensor& indices, const tensor& element_shape, datatype element_dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListGather", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, element_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_get_item(const tensor& input_handle, const tensor& index, const tensor& element_shape, datatype element_dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListGetItem", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, element_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_length(const tensor& input_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListLength", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_push_back(const tensor& input_handle, const tensor& input_tensor, datatype element_dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListPushBack", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_push_back_batch(const tensor& input_handles, const tensor& input_tensor, datatype element_dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListPushBackBatch", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handles.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_reserve(const tensor& element_shape, const tensor& num_elements, datatype element_dtype, datatype shape_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListReserve", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, element_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_elements.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);
    TFE_OpSetAttrType(op, "shape_type", shape_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_resize(const tensor& input_handle, const tensor& size) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListResize", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_scatter(const tensor& input_tensor, const tensor& indices, const tensor& element_shape, datatype element_dtype, datatype shape_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListScatter", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, element_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);
    TFE_OpSetAttrType(op, "shape_type", shape_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_scatter_into_existing_list(const tensor& input_handle, const tensor& input_tensor, const tensor& indices, datatype element_dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListScatterIntoExistingList", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_scatter_v2(const tensor& input_tensor, const tensor& indices, const tensor& element_shape, const tensor& num_elements, datatype element_dtype, datatype shape_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListScatterV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, element_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_elements.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);
    TFE_OpSetAttrType(op, "shape_type", shape_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_set_item(const tensor& input_handle, const tensor& index, const tensor& item, datatype element_dtype) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListSetItem", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, item.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_split(const tensor& input_tensor, const tensor& element_shape, const tensor& lengths, datatype element_dtype, datatype shape_type) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListSplit", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, element_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, lengths.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);
    TFE_OpSetAttrType(op, "shape_type", shape_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_list_stack(const tensor& input_handle, const tensor& element_shape, datatype element_dtype, int64_t num_elements=-1) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorListStack", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, element_shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "element_dtype", element_dtype);
    TFE_OpSetAttrInt(op, "num_elements", num_elements);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_scatter_add(const tensor& input_tensor, const tensor& indices, const tensor& updates, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorScatterAdd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_scatter_sub(const tensor& input_tensor, const tensor& indices, const tensor& updates, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorScatterSub", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_scatter_update(const tensor& input_tensor, const tensor& indices, const tensor& updates, datatype Tindices) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorScatterUpdate", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, updates.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_slice_dataset(const std::vector<tensor>&components, const std::vector<datatype>& Toutput_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorSliceDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> components_handles; components_handles.reserve((int)components.size());
    std::transform(components.begin(), components.end(), std::back_inserter(components_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, components_handles.data(), (int)components.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "Toutput_types", reinterpret_cast<const enum TF_DataType *>(Toutput_types.data()), (int)Toutput_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_strided_slice_update(const tensor& input, const tensor& begin, const tensor& end, const tensor& strides, const tensor& value, datatype Index, int64_t begin_mask=0, int64_t end_mask=0, int64_t ellipsis_mask=0, int64_t new_axis_mask=0, int64_t shrink_axis_mask=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorStridedSliceUpdate", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, begin.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, end.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, strides.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Index", Index);
    TFE_OpSetAttrInt(op, "begin_mask", begin_mask);
    TFE_OpSetAttrInt(op, "end_mask", end_mask);
    TFE_OpSetAttrInt(op, "ellipsis_mask", ellipsis_mask);
    TFE_OpSetAttrInt(op, "new_axis_mask", new_axis_mask);
    TFE_OpSetAttrInt(op, "shrink_axis_mask", shrink_axis_mask);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_summary(const tensor& input_tensor, const std::vector< std::string>& labels, const std::string& description="", const std::string& display_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorSummary", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    
    std::vector<std::size_t> labels_sizes; labels_sizes.reserve((int)labels.size());
    std::transform(labels.begin(), labels.end(), std::back_inserter(labels_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "labels", reinterpret_cast<const void *const *>(labels.data()), labels_sizes.data(), (int)labels.size());
    
    TFE_OpSetAttrString(op, "description", (void*) description.c_str(), description.size());
    TFE_OpSetAttrString(op, "display_name", (void*) display_name.c_str(), display_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tensor_summary_v2(const tensor& tag, const tensor& input_tensor, const tensor& serialized_summary_metadata) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TensorSummaryV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, tag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, serialized_summary_metadata.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor text_line_dataset(const tensor& filenames, const tensor& compression_type, const tensor& buffer_size) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TextLineDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, filenames.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, compression_type.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, buffer_size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor text_line_reader(int64_t skip_header_lines=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TextLineReader", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrInt(op, "skip_header_lines", skip_header_lines);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor text_line_reader_v2(int64_t skip_header_lines=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TextLineReaderV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrInt(op, "skip_header_lines", skip_header_lines);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor thread_pool_dataset(const tensor& input_dataset, const tensor& thread_pool, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ThreadPoolDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, thread_pool.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor thread_pool_handle(int64_t num_threads, const std::string& display_name, int64_t max_intra_op_parallelism=1, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ThreadPoolHandle", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrInt(op, "num_threads", num_threads);
    TFE_OpSetAttrString(op, "display_name", (void*) display_name.c_str(), display_name.size());
    TFE_OpSetAttrInt(op, "max_intra_op_parallelism", max_intra_op_parallelism);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tile(const tensor& input, const tensor& multiples, datatype Tmultiples=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Tile", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, multiples.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tmultiples", Tmultiples);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tile_grad(const tensor& input, const tensor& multiples) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TileGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, multiples.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor timestamp() {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Timestamp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor to_bool(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ToBool", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor transpose(const tensor& x, const tensor& perm, datatype Tperm=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Transpose", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, perm.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tperm", Tperm);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tridiagonal_mat_mul(const tensor& superdiag, const tensor& maindiag, const tensor& subdiag, const tensor& rhs) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TridiagonalMatMul", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, superdiag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, maindiag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, subdiag.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rhs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor tridiagonal_solve(const tensor& diagonals, const tensor& rhs, bool partial_pivoting=true) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TridiagonalSolve", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, diagonals.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, rhs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrBool(op, "partial_pivoting", (unsigned char)partial_pivoting);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor truncate_div(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TruncateDiv", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor truncate_mod(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TruncateMod", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor truncated_normal(const tensor& shape, datatype dtype, int64_t seed=0, int64_t seed2=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "TruncatedNormal", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, shape.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrInt(op, "seed", seed);
    TFE_OpSetAttrInt(op, "seed2", seed2);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unbatch(const tensor& batched_input_tensor, const tensor& batch_index, const tensor& id, int64_t timeout_micros, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Unbatch", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, batched_input_tensor.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, batch_index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, id.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "timeout_micros", timeout_micros);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unbatch_dataset(const tensor& input_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnbatchDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unbatch_grad(const tensor& original_input, const tensor& batch_index, const tensor& grad, const tensor& id, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnbatchGrad", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, original_input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, batch_index.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, grad.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, id.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unicode_encode(const tensor& input_values, const tensor& input_splits, const std::string& output_encoding, const std::string& errors="replace", int64_t replacement_char=65533, datatype Tsplits=static_cast<datatype>(9)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnicodeEncode", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, input_splits.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "output_encoding", (void*) output_encoding.c_str(), output_encoding.size());
    TFE_OpSetAttrString(op, "errors", (void*) errors.c_str(), errors.size());
    TFE_OpSetAttrInt(op, "replacement_char", replacement_char);
    TFE_OpSetAttrType(op, "Tsplits", Tsplits);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unicode_script(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnicodeScript", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unicode_transcode(const tensor& input, const std::string& input_encoding, const std::string& output_encoding, const std::string& errors="replace", int64_t replacement_char=65533, bool replace_control_characters=false) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnicodeTranscode", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrString(op, "input_encoding", (void*) input_encoding.c_str(), input_encoding.size());
    TFE_OpSetAttrString(op, "output_encoding", (void*) output_encoding.c_str(), output_encoding.size());
    TFE_OpSetAttrString(op, "errors", (void*) errors.c_str(), errors.size());
    TFE_OpSetAttrInt(op, "replacement_char", replacement_char);
    TFE_OpSetAttrBool(op, "replace_control_characters", (unsigned char)replace_control_characters);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unique_dataset(const tensor& input_dataset, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UniqueDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unpack(const tensor& value, int64_t num, int64_t axis=0) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Unpack", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, value.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrInt(op, "num", num);
    TFE_OpSetAttrInt(op, "axis", axis);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unravel_index(const tensor& indices, const tensor& dims, datatype Tidx=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnravelIndex", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, indices.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, dims.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tidx", Tidx);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unsorted_segment_join(const tensor& inputs, const tensor& segment_ids, const tensor& num_segments, datatype Tindices, const std::string& separator="", datatype Tnumsegments=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnsortedSegmentJoin", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, inputs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_segments.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrString(op, "separator", (void*) separator.c_str(), separator.size());
    TFE_OpSetAttrType(op, "Tnumsegments", Tnumsegments);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unsorted_segment_max(const tensor& data, const tensor& segment_ids, const tensor& num_segments, datatype Tindices, datatype Tnumsegments=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnsortedSegmentMax", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_segments.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrType(op, "Tnumsegments", Tnumsegments);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unsorted_segment_min(const tensor& data, const tensor& segment_ids, const tensor& num_segments, datatype Tindices, datatype Tnumsegments=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnsortedSegmentMin", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_segments.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrType(op, "Tnumsegments", Tnumsegments);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unsorted_segment_prod(const tensor& data, const tensor& segment_ids, const tensor& num_segments, datatype Tindices, datatype Tnumsegments=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnsortedSegmentProd", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_segments.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrType(op, "Tnumsegments", Tnumsegments);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unsorted_segment_sum(const tensor& data, const tensor& segment_ids, const tensor& num_segments, datatype Tindices, datatype Tnumsegments=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnsortedSegmentSum", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, data.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, segment_ids.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, num_segments.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "Tindices", Tindices);
    TFE_OpSetAttrType(op, "Tnumsegments", Tnumsegments);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unstage(const std::vector<datatype>& dtypes, int64_t capacity=0, int64_t memory_limit=0, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Unstage", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "dtypes", reinterpret_cast<const enum TF_DataType *>(dtypes.data()), (int)dtypes.size());
    TFE_OpSetAttrInt(op, "capacity", capacity);
    TFE_OpSetAttrInt(op, "memory_limit", memory_limit);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor unwrap_dataset_variant(const tensor& input_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UnwrapDatasetVariant", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor upper_bound(const tensor& sorted_inputs, const tensor& values, datatype out_type=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "UpperBound", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, sorted_inputs.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, values.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor var_handle_op(datatype dtype, const std::vector<int64_t>& shape, const std::vector< std::string>& allowed_devices, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "VarHandleOp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrType(op, "dtype", dtype);
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    
    std::vector<std::size_t> allowed_devices_sizes; allowed_devices_sizes.reserve((int)allowed_devices.size());
    std::transform(allowed_devices.begin(), allowed_devices.end(), std::back_inserter(allowed_devices_sizes), [](const auto& s) { return s.size();});
    TFE_OpSetAttrStringList(op, "allowed_devices", reinterpret_cast<const void *const *>(allowed_devices.data()), allowed_devices_sizes.data(), (int)allowed_devices.size());
    
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor var_is_initialized_op(const tensor& resource) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "VarIsInitializedOp", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, resource.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor variable(const std::vector<int64_t>& shape, datatype dtype, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Variable", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor variable_shape(const tensor& input, datatype out_type=static_cast<datatype>(3)) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "VariableShape", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrType(op, "out_type", out_type);

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor variable_v2(const std::vector<int64_t>& shape, datatype dtype, const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "VariableV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    
    TFE_OpSetAttrShape(op, "shape", shape.data(), (int)shape.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrType(op, "dtype", dtype);
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor where(const tensor& input) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Where", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor whole_file_reader(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "WholeFileReader", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor whole_file_reader_v2(const std::string& container="", const std::string& shared_name="") {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "WholeFileReaderV2", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    

    // Attributes
    TFE_OpSetAttrString(op, "container", (void*) container.c_str(), container.size());
    TFE_OpSetAttrString(op, "shared_name", (void*) shared_name.c_str(), shared_name.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor window_dataset(const tensor& input_dataset, const tensor& size, const tensor& shift, const tensor& stride, const tensor& drop_remainder, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "WindowDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_dataset.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, size.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, shift.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, stride.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, drop_remainder.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor worker_heartbeat(const tensor& request) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "WorkerHeartbeat", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, request.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor wrap_dataset_variant(const tensor& input_handle) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "WrapDatasetVariant", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, input_handle.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor xdivy(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Xdivy", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor xlog1py(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Xlog1py", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor xlogy(const tensor& x, const tensor& y) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Xlogy", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, y.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor zeros_like(const tensor& x) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ZerosLike", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor zeta(const tensor& x, const tensor& q) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "Zeta", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    TFE_OpAddInput(op, x.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    
    
    TFE_OpAddInput(op, q.tfe_handle.get(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


tensor zip_dataset(const std::vector<tensor>&input_datasets, const std::vector<datatype>& output_types, const std::vector< std::vector<int64_t>>& output_shapes) {

    // Define Op
    auto op = TFE_NewOp(context::get_context(), "ZipDataset", context::get_status());
    status_check(context::get_status());

    // Required input arguments
    
    std::vector<TFE_TensorHandle*> input_datasets_handles; input_datasets_handles.reserve((int)input_datasets.size());
    std::transform(input_datasets.begin(), input_datasets.end(), std::back_inserter(input_datasets_handles), [](const auto& t) { return t.tfe_handle.get();});
    TFE_OpAddInputList(op, input_datasets_handles.data(), (int)input_datasets.size(), context::get_status());
    status_check(context::get_status());
    

    // Attributes
    TFE_OpSetAttrTypeList(op, "output_types", reinterpret_cast<const enum TF_DataType *>(output_types.data()), (int)output_types.size());
    
    std::vector<const int64_t*> output_shapes_values; output_shapes_values.reserve((int)output_shapes.size());
    std::vector<int> output_shapes_ndims; output_shapes_ndims.reserve((int)output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_values), [](const auto& v) { return v.data();});
    std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shapes_ndims), [](const auto& v) { return (int)v.size();});
    TFE_OpSetAttrShapeList(op, "output_shapes", output_shapes_values.data(), output_shapes_ndims.data(), (int)output_shapes.size(), context::get_status());
    status_check(context::get_status());
    
    TFE_OpSetAttrInt(op, "N", (int)input_datasets.size());

    // Execute Op
    int num_outputs_op = 1;
    TFE_TensorHandle* res[1] = {nullptr};
    TFE_Execute(op, res, &num_outputs_op, context::get_status());
    status_check(context::get_status());
    TFE_DeleteOp(op);
    return tensor(res[0]);
}


} // cppflow

#endif

