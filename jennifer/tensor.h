#ifndef JENNIFER_TENSOR_HPP_
#define JENNIFER_TENSOR_HPP_

#include <vector>
#include <cstdint>
#include <armadillo>

namespace jennifer
{

template <typename T>
class Tensor
{
public:
    Tensor(uint32_t size);
    Tensor(uint32_t rows, uint32_t cols);
    Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
    Tensor(const std::vector<uint32_t>& shapes);

    Tensor(T* data_ptr, uint32_t size);
    Tensor(T* data_ptr, uint32_t rows, uint32_t cols);
    Tensor(T* data_ptr, uint32_t channels, uint32_t rows, uint32_t cols);
    Tensor(T* data_ptr, const std::vector<uint32_t>& shapes);

    uint32_t size() const;
    uint32_t rows() const;
    uint32_t cols() const;
    uint32_t channels() const;

    bool empty() const;
    arma::Cube<T>& get_data();
    const arma::Cube<T>& get_data() const;
    void set_data(const arma::Cube<T>& data);
    std::vector<T> values(bool row_major = true);

    T& at(uint32_t channel, uint32_t row, uint32_t col);
    const T& at(uint32_t channel, uint32_t row, uint32_t col) const;
    T& index(uint32_t offset);
    const T& index(uint32_t offset) const;

    std::vector<uint32_t> shape() const;
    const std::vector<uint32_t> raw_shape() const;

    T* data_ptr();
    T* data_ptr(size_t offset);
    T* matrix_data_ptr(uint32_t index);
    const T* data_ptr() const;
    const T* data_ptr(size_t offset) const;
    const T* matrix_data_ptr(uint32_t index) const;

public:
    arma::Mat<T> Slice(uint32_t channel);
    const arma::Mat<T> Slice(uint32_t channel) const;
    
    void Fill(T value);
    void Fill(const std::vector<T>& values, bool row_major = true);
    void Flatten(bool row_major = false);
    void Padding(const std::vector<uint32_t>& dims, T value);
    
    void Ones();
    void Show();
    void RandomNormal(T mean = 0, T stddev = 1);
    void RandomUniform(T low = 0, T high = 1);

    void Review(const std::vector<uint32_t>& shapes);
    void Reshape(const std::vector<uint32_t>& shapes, bool row_major = false);
    void Transform(const std::function<T(T)>& filter);

private:
    std::vector<uint32_t> shape_;
    arma::Cube<T> data_;
};

} // namespace jennifer

#endif // JENNIFER_TENSOR_HPP_