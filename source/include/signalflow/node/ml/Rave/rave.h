#pragma once
#include "signalflow/node/node.h"
#include "signalflow/node/ml/Rave/backend.h"
#include "signalflow/node/ml/Rave/circular_buffer.h"

#include <vector>
#include <string>

namespace signalflow
{
/**--------------------------------------------------------------------------------*
 * Rave model generator
 *---------------------------------------------------------------------------------*/
class Rave : public Node
{
public:
    Rave(
        std::string path = "",
        std::string method = "forward",
        int buffer_size = 4096
    );

    virtual void alloc() override;
    virtual void process(Buffer &out, int num_frames) override;

private:
    bool load_model(std::string path);
    void model_perform();

    bool m_enabled = false;

    // BACKEND RELATED MEMBERS
    std::unique_ptr<::nn_tilde::Backend> m_model;
    std::vector<std::string> settable_attributes;
    std::string m_path;
    std::string m_method;

    // BUFFER RELATED MEMBERS
    int m_in_dim, m_in_ratio, m_out_dim, m_out_ratio, m_buffer_size;

    std::unique_ptr<::nn_tilde::circular_buffer<float, float>[]> m_in_buffer;
    std::unique_ptr<::nn_tilde::circular_buffer<float, float>[]> m_out_buffer;
    std::vector<std::unique_ptr<float[]>> m_in_model, m_out_model;

    // TODO Use input samples and remove this
    std::vector<float> input_values_mock;
};

REGISTER(Rave, "rave")
}
