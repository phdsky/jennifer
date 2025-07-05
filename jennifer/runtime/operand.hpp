#ifndef JENNIFER_RUNTIME_OPERAND_HPP
#define JENNIFER_RUNTIME_OPERAND_HPP

#include <cstddef>
#include <string>

#include "jennifer/data/tensor.hpp"

#include "attribute.hpp"

namespace jennifer
{
namespace runtime
{

using namespace data;

template <typename T>
struct Operand;

template <>
struct Operand<float>
{
    explicit Operand() = default;

    explicit Operand(std::string name, std::vector<int32_t> shapes,
                     std::vector<std::shared_ptr<Tensor<float>>> data, AttributeType type) :
        name(std::move(name)), shapes(std::move(shapes)), data(std::move(data)), type(type)
    {
    }

    explicit Operand(std::string name, std::vector<int32_t> shapes,
                     uint32_t data_size, AttributeType type) :
        name(std::move(name)), shapes(std::move(shapes)), type(type)
    {
        data.resize(data_size);
    }

    size_t size() const;

    std::string name;

    std::vector<int32_t> shapes;

    std::vector<std::shared_ptr<Tensor<float>>> data;

    AttributeType type = AttributeType::Unknown;
}; // struct Operand

inline size_t Operand<float>::size() const
{
    if (shapes.empty())
    {
        return 0;
    }

    size_t size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<>());
    return size;
}

// using Operand = Operand<float>;
using OperandQuantized = Operand<int8_t>;

} // namespace runtime
} // namespace jennifer

#endif // JENNIFER_RUNTIME_OPERAND_HPP
