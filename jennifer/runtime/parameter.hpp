#ifndef JENNIFER_RUNTIME_PARAMETER_HPP
#define JENNIFER_RUNTIME_PARAMETER_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace jennifer
{
namespace runtime
{

enum class ParameterType
{
    Unknown = 0,
    Bool = 1,
    Int = 2,
    Float = 3,
    String = 4,
    IntArray = 5,
    FloatArray = 6,
    StringArray = 7,
}; // enum class ParameterType

struct Parameter
{
    virtual ~Parameter() = default;

    explicit Parameter(ParameterType type = ParameterType::Unknown) :
        type(type)
    {
    }

    ParameterType type = ParameterType::Unknown;
};

struct ParameterInt : public Parameter
{
    explicit ParameterInt() :
        Parameter(ParameterType::Int)
    {
    }

    explicit ParameterInt(int32_t param_value) :
        Parameter(ParameterType::Int), value(param_value)
    {
    }

    int32_t value = 0;
};

struct ParameterFloat : public Parameter
{
    explicit ParameterFloat() :
        Parameter(ParameterType::Float)
    {
    }

    explicit ParameterFloat(float param_value) :
        Parameter(ParameterType::Float), value(param_value)
    {
    }

    float value = 0.f;
};

struct ParameterString : public Parameter
{
    explicit ParameterString() :
        Parameter(ParameterType::String)
    {
    }

    explicit ParameterString(std::string param_value) :
        Parameter(ParameterType::String), value(std::move(param_value))
    {
    }

    std::string value;
};

struct ParameterIntArray : public Parameter
{
    explicit ParameterIntArray() :
        Parameter(ParameterType::IntArray)
    {
    }

    explicit ParameterIntArray(std::vector<int32_t> param_value) :
        Parameter(ParameterType::IntArray), value(std::move(param_value))
    {
    }

    std::vector<int32_t> value;
};

struct ParameterFloatArray : public Parameter
{
    explicit ParameterFloatArray() :
        Parameter(ParameterType::FloatArray)
    {
    }

    explicit ParameterFloatArray(std::vector<float> param_value) :
        Parameter(ParameterType::FloatArray), value(std::move(param_value))
    {
    }

    std::vector<float> value;
};

struct ParameterStringArray : public Parameter
{
    explicit ParameterStringArray() :
        Parameter(ParameterType::StringArray)
    {
    }

    explicit ParameterStringArray(std::vector<std::string> param_value) :
        Parameter(ParameterType::StringArray), value(std::move(param_value))
    {
    }

    std::vector<std::string> value;
};

struct ParameterBool : public Parameter
{
    explicit ParameterBool() :
        Parameter(ParameterType::Bool)
    {
    }

    explicit ParameterBool(bool param_value) :
        Parameter(ParameterType::Bool), value(param_value)
    {
    }

    bool value = false;
};

} // namespace runtime
} // namespace jennifer

#endif // JENNIFER_RUNTIME_PARAMETER_HPP
