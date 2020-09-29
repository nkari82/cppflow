//
// Created by serizba on 29/6/20.
//

#ifndef CPPFLOW2_MODEL_H
#define CPPFLOW2_MODEL_H

#include <tensorflow/c/c_api.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>

#include "context.h"
#include "tensor.h"

namespace cppflow {

	class model {
	public:
		class operation
		{
		public:
			operation(TF_Graph* graph, TF_Operation* op);
			const std::string& name();
			const int32_t dimention();

		private:
			std::string name_;
			int32_t n_dims_;
			TF_Operation* op_;
		};

		class operations
		{
		public:
			class iterator
			{
			public:
				iterator() = default;
				iterator(TF_Graph* graph, size_t pos);
				iterator operator++();
				operation operator*();
				bool operator!=(const iterator& iter);
				bool operator==(const iterator& iter);

			private:
				TF_Graph* graph_;
				size_t pos_;
				TF_Operation* op_;
			};

			operations(model* _model);

			iterator begin();
			iterator end();

		private:
			TF_Graph* graph_;
		};

		explicit model(const std::string& filename
			, const std::vector<std::string>& inputs = { "serving_default_input_1" }
			, const std::vector<std::string>& outputs = { "StatefulPartitionedCall" });

		tensor operator()(const tensor& input);
		void operator()(const std::vector<tensor>& inputs, std::vector<tensor>& outputs);

	private:
		TF_Graph* graph;
		TF_Session* session;
		std::vector<TF_Output> inputs;
		std::vector<TF_Output> outputs;
	};
}

namespace cppflow {
	model::model(const std::string& filename, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs) {
		this->graph = TF_NewGraph();

		// Create the session.
		TF_SessionOptions* session_options = TF_NewSessionOptions();
		TF_Buffer* run_options = TF_NewBufferFromString("", 0);
		TF_Buffer* meta_graph = TF_NewBuffer();

		int tag_len = 1;
		const char* tag = "serve";
		this->session = TF_LoadSessionFromSavedModel(session_options, run_options, filename.c_str(), &tag, tag_len, graph, meta_graph, context::get_status());
		TF_DeleteSessionOptions(session_options);
		TF_DeleteBuffer(run_options);
		TF_DeleteBuffer(meta_graph);

		status_check(context::get_status());

		for (size_t i = 0; i < inputs.size(); ++i)
		{
			TF_Operation* oper = TF_GraphOperationByName(this->graph, inputs[i].c_str());
			if (oper == nullptr) throw std::runtime_error{ "invalid operator!" };
			this->inputs.emplace(this->inputs.end(), TF_Output{ oper, 0 });
		}

		for (size_t i = 0; i < outputs.size(); ++i)
		{
			TF_Operation* oper = TF_GraphOperationByName(this->graph, outputs[i].c_str());
			if (oper == nullptr) throw std::runtime_error{ "invalid operator!" };
			this->outputs.emplace(this->outputs.end(), TF_Output{ oper, 0 });
		}
	}

	tensor model::operator()(const tensor& input) {
		assert(inputs.size() == outputs.size() == 1);

		//********* Allocate data for inputs & outputs
		auto inp_tensor = TFE_TensorHandleResolve(input.tfe_handle.get(), context::get_status());
		status_check(context::get_status());

		TF_Tensor* inpvals[1] = { inp_tensor };
		TF_Tensor* outvals[1] = { nullptr };

		TF_SessionRun(this->session
			, nullptr
			, this->inputs.data(), inpvals, (int)this->inputs.size()
			, this->outputs.data(), outvals, (int)this->outputs.size()
			, nullptr, 0, nullptr, context::get_status());

		status_check(context::get_status());

		return tensor(outvals[0]);
	}

	void model::operator()(const std::vector<tensor>& inputs, std::vector<tensor>& outputs)
	{
		try
		{
			//********* Allocate data for inputs & outputs
			std::vector<TF_Tensor*> inpvals;
			std::vector<TF_Tensor*> outvals;

			inpvals.reserve(inputs.size());
			outvals.resize(this->outputs.size());

			for (auto& v : inputs)
				inpvals.emplace_back(TFE_TensorHandleResolve(v.tfe_handle.get(), context::get_status()));

			status_check(context::get_status());

			TF_SessionRun(this->session
				, nullptr
				, this->inputs.data(), inpvals.data(), (int)this->inputs.size()
				, this->outputs.data(), outvals.data(), (int)this->outputs.size()
				, nullptr, 0, nullptr, context::get_status());

			status_check(context::get_status());

			for (auto& v : outvals)
				outputs.emplace_back(tensor{ v });
		}
		catch (const std::exception& ex)
		{
			std::cerr << ex.what() << std::endl;
			throw ex;
		}
	}

	model::operation::operation(TF_Graph* graph, TF_Operation* op) : op_(op), n_dims_(0)
	{
		name_ = TF_OperationName(op_);
		if (std::strcmp("NoOp", name_.c_str()) == 0)
			return;

		n_dims_ = TF_GraphGetTensorNumDims(graph, { op_, 0 }, context::get_status());
	}

	const std::string& model::operation::name()
	{
		return name_;
	}

	const int32_t model::operation::dimention()
	{
		return n_dims_;
	}

	model::operations::iterator::iterator(TF_Graph* graph, size_t pos)
		: graph_(graph)
		, pos_(pos)
		, op_(nullptr)
	{
		if (pos != std::numeric_limits<size_t>::max())
			op_ = TF_GraphNextOperation(graph, &pos_);
	}

	model::operations::iterator model::operations::iterator::operator++()
	{
		op_ = TF_GraphNextOperation(graph_, &pos_);
		return *this;
	}

	model::operation model::operations::iterator::operator*()
	{
		return operation(graph_, op_);
	}

	bool model::operations::iterator::operator!=(const iterator& iter)
	{
		return bool(op_ != iter.op_);
	}

	bool model::operations::iterator::operator==(const iterator& iter)
	{
		return bool(op_ == iter.op_);
	}

	model::operations::operations(model* _model) : graph_(_model->graph)
	{}

	model::operations::iterator model::operations::begin()
	{
		return operations::iterator(graph_, (size_t)0);
	}

	model::operations::iterator model::operations::end()
	{
		return operations::iterator(graph_, std::numeric_limits<size_t>::max());
	}
}

#endif //CPPFLOW2_MODEL_H
