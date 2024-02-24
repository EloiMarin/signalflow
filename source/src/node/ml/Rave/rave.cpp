#include "signalflow/node/ml/Rave/rave.h"
#include "signalflow/node/ml/Rave/circular_buffer.h"
#include "signalflow/node/ml/Rave/utils.h"
#include "signalflow/core/core.h"

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>

namespace signalflow
{

Rave::Rave(std::string path, std::string method, int buffer_size)
    : m_buffer_size(buffer_size)
{
    SIGNALFLOW_CHECK_GRAPH();

    this->name = "rave";

    m_model = std::make_unique<::nn_tilde::Backend>();
    m_in_dim = 1;
    m_in_ratio = 1;
    m_out_dim = 1;
    m_out_ratio = 1;

    // TODO Create inputs
    // this->create_input("path", this->path);

    if (this->load_model(path, method)) {
        this->alloc();

        // HACK1 BEGINS
        // This whole thing underneath is to warm-up the caches until
        // perform_model can run within reasonable time limits.
        // Probably there is a way better solution waiting to be implemented.
        const double runtime_limit = 30.0;
        const int min = 5;
        const int max = 64;
        int good_performance_count = 0;
        double best_interval = 100000.0;
        for (int i = 0; i < max; ++i) {
            using std::chrono::high_resolution_clock;
            using std::chrono::duration;
            using std::chrono::milliseconds;

            auto t1 = high_resolution_clock::now();
            model_perform();
            auto t2 = high_resolution_clock::now();
            duration<double, std::milli> interval = t2 - t1;

            if (interval.count() < best_interval) {
                best_interval = interval.count();
            }

            if (
                interval.count() < runtime_limit &&
                ++good_performance_count >= min
            ) break;
        }

        if (good_performance_count >= min) {
            m_enabled = true;
        } else {
            std::stringstream warning_message;
            warning_message << "[Rave] Could not achieve real-time performance: " << best_interval;
            signalflow_warn(warning_message.str().c_str());
        }
        // HACK1 ENDS
    }
}

bool Rave::load_model(std::string path, std::string method) {
    try {
        std::string loading_message = "[Rave] Loading: ";
        loading_message += path;
        signalflow_debug(loading_message.c_str());

        int load_result = m_model->load(path, method);

        if (load_result == -1) {
            signalflow_warn("[Rave] Could not load model");
            return false;
        } else if (load_result == 1) {
            std::stringstream warning_message;
            warning_message << "[Rave] Method does not exist in model: " << method;
            signalflow_warn(warning_message.str().c_str());
            return false;
        }
        signalflow_debug("[Rave] Model loaded");

        // GET MODEL'S METHOD PARAMETERS
        auto params = m_model->get_method_params(method);
        settable_attributes = m_model->get_settable_attributes();

        m_in_dim = params[0];
        m_in_ratio = params[1];
        m_out_dim = params[2];
        m_out_ratio = params[3];

        auto higher_ratio = m_model->get_higher_ratio();

        if (!m_buffer_size) {
            m_buffer_size = higher_ratio;
        } else if (m_buffer_size < higher_ratio) {
            m_buffer_size = higher_ratio;

            std::string err_message = "[Rave] buffer size too small, switching to ";
            err_message += std::to_string(higher_ratio);
            signalflow_warn(err_message.c_str());
        } else {
            m_buffer_size = power_ceil(m_buffer_size);
        }
        return true;
    } catch (const std::exception &e) {
        signalflow_warn(e.what());
        return false;
    }
}

void Rave::alloc()
{
    input_values_mock = std::vector<float>(m_buffer_size, 0.0f);
    m_in_buffer = std::make_unique<::nn_tilde::circular_buffer<float, float>[]>(m_in_dim);

    for (int in_dimension = 0; in_dimension < m_in_dim; in_dimension++) {
        if (in_dimension < m_in_dim - 1) { // Is this check related to some Pd shenanigan? Remove if so
            // TODO Create input signal inlets for each of the input dimensions
            //this->create_input("path", this->path);
        }
        m_in_buffer[in_dimension].initialize(m_buffer_size);
        m_in_model.push_back(std::make_unique<float[]>(m_buffer_size));
    }

    // TODO Use this->num_output_channels and handle when the model has different output dimensions than the Node's number of output channels
    m_out_buffer = std::make_unique<::nn_tilde::circular_buffer<float, float>[]>(m_out_dim);
    for (int i(0); i < m_out_dim; i++) {
        m_out_buffer[i].initialize(m_buffer_size);
        m_out_model.push_back(std::make_unique<float[]>(m_buffer_size));
    }
}

void Rave::process(Buffer &out, int num_frames)
{
    // If the model is not loaded or not ready, just output silence
    if (!m_model->is_loaded() || !m_enabled) {
        for (int channel = 0; channel < m_out_dim; channel++)
            for (int frame = 0; frame < num_frames; frame++)
                out[channel][frame] = 0.0f;
        return;
    }

    // COPY INPUT TO CIRCULAR BUFFER
    for (int in_dim = 0; in_dim < m_in_dim; in_dim++) {
        // TODO Use input samples
        //m_in_buffer[in_dim].put(m_dsp_in_vec[in_dim], num_frames);
        m_in_buffer[in_dim].put(&input_values_mock[0], num_frames);
    }

    if (m_in_buffer[0].full()) {
        // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
        for (int in_dimension = 0; in_dimension < m_in_dim; in_dimension++)
            m_in_buffer[in_dimension].get(
                m_in_model[in_dimension].get(),
                m_buffer_size
            );

        model_perform();

        // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
        for (int out_dimension = 0; out_dimension < m_out_dim; out_dimension++)
            m_out_buffer[out_dimension].put(
                m_out_model[out_dimension].get(),
                m_buffer_size
            );
    }

    // COPY CIRCULAR BUFFER TO OUTPUT
    auto channels = m_out_dim > this->num_output_channels
         ? this->num_output_channels
         : m_out_dim;
    for (int channel = 0; channel < channels; channel++)
        m_out_buffer[channel].get(out[channel], num_frames);
}

void Rave::model_perform() {
    std::vector<float *> in_model, out_model;

    for (int c(0); c < m_in_dim; c++)
        in_model.push_back(m_in_model[c].get());
    for (int c(0); c < m_out_dim; c++)
        out_model.push_back(m_out_model[c].get());

    m_model->perform(in_model, out_model, m_buffer_size, 1);
}

/* TODO Set model attributes
void nn_tilde_set(t_nn_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc < 2) {
        signalflow_warn("set needs at least 2 arguments [set argname argval1 ...)");
        return;
    }
    std::vector<std::string> attribute_args;

    auto argname = argv[0].a_w.w_symbol->s_name;
    std::string argname_str = argname;

    if (!std::count(x->settable_attributes.begin(), x->settable_attributes.end(), argname_str)) {
        signalflow_warn("argument name not settable in current model");
        return;
    }

    for (int i(1); i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            attribute_args.push_back(argv[i].a_w.w_symbol->s_name);
        } else if (argv[i].a_type == A_FLOAT) {
            attribute_args.push_back(std::to_string(argv[i].a_w.w_float));
        }
    }
    try {
        x->m_model->set_attribute(argname, attribute_args);
    } catch (const std::exception &e) {
        signalflow_warn(e.what());
    }
}
*/

// TODO Get model attributes. This is missing in the nn~ object, but I wonder... how can I know which attributes could I set if I can't query them?
}
