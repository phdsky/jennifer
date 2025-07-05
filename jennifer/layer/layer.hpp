#ifndef JENNIFER_LAYER_LAYER_HPP_
#define JENNIFER_LAYER_LAYER_HPP_

#include <string>

namespace jennifer
{
namespace layer
{

template <typename T>
class Layer;

template <>
class Layer<float>
{
public:
protected:
    std::string layer_name;

}; // class Layer

} // namespace layer
} // namespace jennifer

#endif // JENNIFER_LAYER_LAYER_HPP_
