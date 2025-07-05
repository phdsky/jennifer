#ifndef JENNIFER_UTILS_COMMON_HPP
#define JENNIFER_UTILS_COMMON_HPP

namespace jennifer
{
namespace utils
{

enum class StatusCode
{
    Unknown = -1,
    Success = 0,

    InferInputsEmpty = 1,
    InferOutputsEmpty = 2,
    InferParamError = 3,
    InferDimMismatch = 4,

    FunctionNotImplement = 5,
    ParseWeightError = 6,
    ParseParamError = 7,
    ParseNullOperator = 8,
}; // enum class StatusCode

} // namespace utils
} // namespace jennifer

#endif // JENNIFER_UTILS_COMMON_HPP
