#ifndef JENNIFER_RUNTIME_OPERATOR_HPP
#define JENNIFER_RUNTIME_OPERATOR_HPP

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include "operand.hpp"
#include "parameter.hpp"

namespace jennifer
{
namespace runtime
{

template <typename T>
class Layer;

template <typename T>
struct Operator
{
    std::string name;
    std::string type;

    int32_t start_time = -1;
    int32_t end_time = -1;
    int32_t occur_end_time = -1;

    bool has_forward = false;

    std::shared_ptr<Layer<T>> layer;

    std::vector<std::string> output_names;

    std::shared_ptr<Operand<T>> output_operands;
    std::map<std::string, std::shared_ptr<Operand<T>>> input_operands;
    std::vector<std::shared_ptr<Operand<T>>> input_operands_seq;

    std::map<std::string, std::shared_ptr<Operator<T>>> output_operators;

}; // struct Operator

} // namespace runtime
} // namespace jennifer

#endif // JENNIFER_RUNTIME_OPERATOR_HPP
