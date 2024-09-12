#include <glog/logging.h>

#include "jennifer/tensor.h"

namespace jennifer
{

template<typename T>
Tensor<T>::Tensor(uint32_t size)
{
    data_ = arma::Cube<T>(1, size, 1);
    shape_ = {size};
}

template<typename T>
Tensor<T>::Tensor(uint32_t rows, uint32_t cols)
{
    data_ = arma::Cube<T>(rows, cols, 1);
    if (rows == 1) {
        shape_ = {cols};
    } else {
        shape_ = {rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols)
{
    data_ = arma::Cube<T>(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        shape_ = {cols};
    } else if (channels == 1) {
        shape_ = {rows, cols};
    } else {
        shape_ = {channels, rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shapes)
{
    CHECK(!shapes.empty() && shapes.size() <= 3);

    uint32_t remain = 3 - shapes.size();
    std::vector<uint32_t> new_shapes(3, 1);
    std::copy(shapes.begin(), shapes.end(), new_shapes.begin() + remain);

    uint32_t channels = new_shapes[0];
    uint32_t rows = new_shapes[1];
    uint32_t cols = new_shapes[2];

    data_ = arma::Cube<T>(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        shape_ = {cols};
    } else if (channels == 1) {
        shape_ = {rows, cols};
    } else {
        shape_ = {channels, rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(T* data_ptr, uint32_t size)
{
    CHECK_NE(data_ptr, nullptr);
    data_ = arma::Cube<T>(data_ptr, 1, size, 1, false, true);
    shape_ = {size};
}

template<typename T>
Tensor<T>::Tensor(T* data_ptr, uint32_t rows, uint32_t cols)
{
    CHECK_NE(data_ptr, nullptr);
    data_ = arma::Cube<T>(data_ptr, rows, cols, 1, false, true);
    if (rows == 1) {
        shape_ = {cols};
    } else {
        shape_ = {rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(T* data_ptr, uint32_t channels, uint32_t rows, uint32_t cols)
{
    CHECK_NE(data_ptr, nullptr);
    data_ = arma::Cube<T>(data_ptr, rows, cols, channels, false, true);
    if (channels == 1 && rows == 1) {
        shape_ = {cols};
    } else if (channels == 1) {
        shape_ = {rows, cols};
    } else {
        shape_ = {channels, rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(T* data_ptr, const std::vector<uint32_t>& shapes)
{
    CHECK_NE(data_ptr, nullptr);
    CHECK(!shapes.empty() && shapes.size() <= 3);

    uint32_t remain = 3 - shapes.size();
    std::vector<uint32_t> new_shapes(3, 1);
    std::copy(shapes.begin(), shapes.end(), new_shapes.begin() + remain);

    uint32_t channels = new_shapes[0];
    uint32_t rows = new_shapes[1];
    uint32_t cols = new_shapes[2];

    data_ = arma::Cube<T>(data_ptr, rows, cols, channels, false, true);
    if (channels == 1 && rows == 1) {
        shape_ = {cols};
    } else if (channels == 1) {
        shape_ = {rows, cols};
    } else {
        shape_ = {channels, rows, cols};
    }
}

template<typename T>
uint32_t Tensor<T>::size() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return data_.size();
}

template<typename T>
uint32_t Tensor<T>::rows() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return data_.n_rows;
}

template<typename T>
uint32_t Tensor<T>::cols() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return data_.n_cols;
}

template<typename T>
uint32_t Tensor<T>::channels() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return data_.n_slices;
}

// bool empty() const;
// arma::Cube<T>& get_data();
// const arma::Cube<T>& get_data() const;
// void set_data(const arma::Cube<T>& data);
// std::vector<T> values(bool row_major = true);

// T& at(uint32_t channel, uint32_t row, uint32_t col);
// const T& at(uint32_t channel, uint32_t row, uint32_t col) const;
// T& index(uint32_t offset);
// const T& index(uint32_t offset) const;

// std::vector<uint32_t> shape() const;
// const std::vector<uint32_t> raw_shape() const;

// T* data_ptr();
// T* data_ptr(size_t offset);
// T* matrix_data_ptr(uint32_t index);
// const T* data_ptr() const;
// const T* data_ptr(size_t offset) const;
// const T* matrix_data_ptr(uint32_t index) const;


} // namespace jeenifer