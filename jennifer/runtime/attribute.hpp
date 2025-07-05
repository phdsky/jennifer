#ifndef JENNIFER_RUNTIME_ATTRIBUTE_HPP
#define JENNIFER_RUNTIME_ATTRIBUTE_HPP

#include <type_traits>
#include <vector>
#include <cstdint>
#include <utility>

#include <glog/logging.h>

namespace jennifer
{
namespace runtime
{

enum class AttributeType
{
    Unknown = 0,
    Float32 = 1,
    Float64 = 2,
    Float16 = 3,
    Int32 = 4,
    Int64 = 5,
    Int16 = 6,
    Int8 = 7,
    UInt8 = 8,
}; // enum AttributeType

struct Attribute
{
    Attribute() = default;

    explicit Attribute(std::vector<int32_t> shape, std::vector<char> weight, AttributeType type) :
        shape(std::move(shape)), weight(std::move(weight)), type(type)
    {
    }

    std::vector<int32_t> shape;

    std::vector<char> weight;

    AttributeType type = AttributeType::Unknown;

    template <typename T>
    std::vector<T> get(bool clear_weight = true);

}; // struct Attribute

template <typename T>
std::vector<T> Attribute::get(bool clear_weight)
{
    CHECK(!weight.empty());
    CHECK(type != AttributeType::Unknown);

    const auto elem_size = sizeof(T);
    CHECK_EQ(weight.size() % elem_size, 0);
    const auto weight_size = weight.size() / elem_size;

    std::vector<T> data;
    data.resize(weight_size);
    switch (type)
    {
    case AttributeType::Float32: {
        static_assert(std::is_same<T, float>::value == true,
                      "Attribute AttributeType is not float32");
        float *weight_ptr = reinterpret_cast<float *>(weight.data());
        for (size_t i = 0; i < weight_size; ++i)
        {
            data[i] = *(weight_ptr + i);
        }
        break;
    }
    default: {
        static_assert(std::is_same<T, float>::value == false,
                      "Attribute AttributeType is not supported for this type");
        LOG(FATAL) << "Unsupported AttributeType for get: " << static_cast<int>(type);
    }
    }

    if (clear_weight)
    {
        weight.clear();
    }

    return data;
}

} // namespace runtime
} // namespace jennifer

#endif // JENNIFER_RUNTIME_ATTRIBUTE_HPP
