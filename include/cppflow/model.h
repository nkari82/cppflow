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
        explicit model(const std::string& filename
            , const std::vector<std::string>& inputs = { "serving_default_input_1" }
            , const std::vector<std::string>& outputs = { "StatefulPartitionedCall" });

        std::vector<std::string> get_operations() const;

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
        //TF_DeleteBuffer(meta_graph);

        status_check(context::get_status());

		for (size_t i = 0; i < inputs.size(); ++i)
		{
			TF_Operation* oper = TF_GraphOperationByName(this->graph, inputs[i].c_str());
            if (oper == nullptr) break;

			this->inputs.emplace_back(TF_Output{ oper, (int)i });
		}

		for (size_t i = 0; i < outputs.size(); ++i)
		{
			TF_Operation* oper = TF_GraphOperationByName(this->graph, outputs[i].c_str());
            if (oper == nullptr) break;

			this->outputs.emplace_back(TF_Output{ oper, (int)i });
		}
    }

    std::vector<std::string> model::get_operations() const {
        std::vector<std::string> result;
        size_t pos = 0;
        TF_Operation* oper;

        // Iterate through the operations of a graph
        while ((oper = TF_GraphNextOperation(this->graph, &pos)) != nullptr) {
            result.emplace_back(TF_OperationName(oper));
        }
        return result;
    }

	tensor model::operator()(const tensor& input) {
        assert(inputs.size() == outputs.size() == 1);

        //********* Allocate data for inputs & outputs
        auto inp_tensor = TFE_TensorHandleResolve(input.tfe_handle.get(), context::get_status());
        status_check(context::get_status());

        TF_Tensor* inpvals[1] = {inp_tensor};
        TF_Tensor* outvals[1] = {nullptr};

        TF_SessionRun(this->session
            , NULL
            , this->inputs.data(), inpvals, (int)this->inputs.size()
            , this->outputs.data(), outvals, (int)this->outputs.size()
            , NULL, 0, NULL, context::get_status());

        status_check(context::get_status());

        return tensor(outvals[0]);
    }

    void model::operator()(const std::vector<tensor>& inputs, std::vector<tensor>& outputs)
    {
        //********* Allocate data for inputs & outputs
        std::vector<TF_Tensor*> inpvals;
        std::vector<TF_Tensor*> outvals;

        inpvals.reserve(inputs.size());
        outvals.resize(outputs.size());

        for (auto& v : inputs)
            inpvals.emplace_back(TFE_TensorHandleResolve(v.tfe_handle.get(), context::get_status()));

        status_check(context::get_status());

        TF_SessionRun(this->session
            , NULL
            , this->inputs.data(), inpvals.data(), (int)this->inputs.size()
            , this->outputs.data(), outvals.data(), (int)this->outputs.size()
            , NULL, 0, NULL, context::get_status());

        status_check(context::get_status());

        for (auto& v : outvals)
            outputs.emplace_back(tensor{v});
    }
}

#endif //CPPFLOW2_MODEL_H
