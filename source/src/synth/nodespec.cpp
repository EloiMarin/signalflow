#include "signal/synth/nodespec.h"

namespace libsignal
{
NodeSpec::NodeSpec()
    : name("(null)")
{
}

NodeSpec::~NodeSpec()
{
    // free inputs
}

NodeSpec::NodeSpec(std::string name)
    : name(name)
{
}

NodeSpec::NodeSpec(std::string name, float value)
    : name(name)
{
    this->set_value(value);
}

void NodeSpec::set_id(int value)
{
    this->id = value;
}

void NodeSpec::set_name(std::string name)
{
    this->name = name;
}

void NodeSpec::add_input(std::string name, NodeSpec *def)
{
    NodeSpec *def_copy = new NodeSpec();
    *def_copy = *def;
    this->params[name] = def_copy;
}

void NodeSpec::add_input(std::string name, float value)
{
    this->params[name] = new NodeSpec("constant", value);
}

void NodeSpec::set_value(float value)
{
    this->value = value;
    this->is_constant = true;
}

}
