#include <glog/logging.h>

#include "jennifer/tensor.h"

namespace jennifer
{

template <typename T>
Tensor<T>::Tensor(uint32_t size)
{
    data_ = arma::Cube<T>(1, size, 1);
    shape_ = {size};
}

template <typename T>
Tensor<T>::Tensor(uint32_t rows, uint32_t cols)
{
    data_ = arma::Cube<T>(rows, cols, 1);
    if (rows == 1) {
        shape_ = {cols};
    } else {
        shape_ = {rows, cols};
    }
}

template <typename T>
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

template <typename T>
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

template <typename T>
Tensor<T>::Tensor(T* data_ptr, uint32_t size)
{
    CHECK_NE(data_ptr, nullptr);
    data_ = arma::Cube<T>(data_ptr, 1, size, 1, false, true);
    shape_ = {size};
}

template <typename T>
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

template <typename T>
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

template <typename T>
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

template <typename T>
uint32_t Tensor<T>::size() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return data_.size();
}

template <typename T>
uint32_t Tensor<T>::rows() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return data_.n_rows;
}

template <typename T>
uint32_t Tensor<T>::cols() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return data_.n_cols;
}

template <typename T>
uint32_t Tensor<T>::channels() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return data_.n_slices;
}

template <typename T>
bool Tensor<T>::empty() const
{
    return data_.empty();
}

template <typename T>
arma::Cube<T>& Tensor<T>::get_data()
{
    return data_;
}

template <typename T>
const arma::Cube<T>& Tensor<T>::get_data() const
{
    return data_;
}

template <typename T>
void Tensor<T>::set_data(const arma::Cube<T>& data)
{
    CHECK(data.n_rows == data_.n_rows && data.n_cols == data_.n_cols && data.n_slices == data_.n_slices)
        << "Tensor shape mismatch";
    data_ = data;
}

template <typename T>
std::vector<T> Tensor<T>::values(bool row_major)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    std::vector<T> values(data_.size());
    if (!row_major) {
        std::copy(data_.begin(), data_.end(), values.begin());
    } else {
        for (uint32_t i = 0; i < data_.n_slices; ++i) {
            for (uint32_t j = 0; j < data_.n_rows; ++j) {
                for (uint32_t k = 0; k < data_.n_cols; ++k) {
                    values[i * data_.n_rows * data_.n_cols + j * data_.n_cols + k] = data_(j, k, i);
                }
            }
        }
    }
    return values;
}

template <typename T>
T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_LT(channel, data_.n_slices) << "Channel index out of range";
    CHECK_LT(row, data_.n_rows) << "Row index out of range";
    CHECK_LT(col, data_.n_cols) << "Column index out of range";
    return data_(row, col, channel);
}

template <typename T>
const T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_LT(channel, data_.n_slices) << "Channel index out of range";
    CHECK_LT(row, data_.n_rows) << "Row index out of range";
    CHECK_LT(col, data_.n_cols) << "Column index out of range";
    return data_(row, col, channel);
}

template <typename T>
T& Tensor<T>::index(uint32_t offset)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_LT(offset, data_.size()) << "Index out of range";
    return data_(offset);
}

template <typename T>
const T& Tensor<T>::index(uint32_t offset) const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_LT(offset, data_.size()) << "Index out of range";
    return data_(offset);
}

template <typename T>
std::vector<uint32_t> Tensor<T>::shape() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return {data_.n_slices, data_.n_rows, data_.n_cols};
}

template <typename T>
const std::vector<uint32_t> Tensor<T>::raw_shape() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_GE(shape_.size(), 1);
    CHECK_LE(shape_.size(), 3);
    return shape_;
}

template <typename T>
T* Tensor<T>::data_ptr()
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return data_.memptr();
}

template <typename T>
const T* Tensor<T>::data_ptr() const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    return data_.memptr();
}

template <typename T>
T* Tensor<T>::data_ptr(size_t offset)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_LT(offset, data_.size()) << "Index out of range";
    return data_.memptr() + offset;
}

template <typename T>
const T* Tensor<T>::data_ptr(size_t offset) const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_LT(offset, data_.size()) << "Index out of range";
    return data_.memptr() + offset;
}

template <typename T>
T* Tensor<T>::matrix_data_ptr(uint32_t index)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_LT(index, data_.n_slices) << "Channel index out of range";
    return data_.slice(index).memptr();
}

template <typename T>
const T* Tensor<T>::matrix_data_ptr(uint32_t index) const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_LT(index, data_.n_slices) << "Channel index out of range";
    return data_.slice(index).memptr();
}

template <typename T>
arma::Mat<T> Tensor<T>::Slice(uint32_t channel)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_LT(channel, data_.n_slices) << "Channel index out of range";
    return data_.slice(channel);
}

template <typename T>
const arma::Mat<T> Tensor<T>::Slice(uint32_t channel) const
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_LT(channel, data_.n_slices) << "Channel index out of range";
    return data_.slice(channel);
}

template <typename T>
void Tensor<T>::Fill(T value)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    data_.fill(value);
}

template <typename T>
void Tensor<T>::Fill(const std::vector<T>& values, bool row_major)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_EQ(values.size(), data_.size()) << "Values size mismatch";
    if (!row_major) {
        std::copy(values.begin(), values.end(), data_.begin());
    } else {
        for (uint32_t i = 0; i < data_.n_slices; ++i) {
            for (uint32_t j = 0; j < data_.n_rows; ++j) {
                for (uint32_t k = 0; k < data_.n_cols; ++k) {
                    data_(j, k, i) = values[i * data_.n_rows * data_.n_cols + j * data_.n_cols + k];
                }
            }
        }
    }
}

template <typename T>
void Tensor<T>::Flatten(bool row_major)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    const uint32_t size = data_.size();
    this->Reshape({size}, row_major);
}

template <typename T>
void Tensor<T>::Padding(const std::vector<uint32_t>& dims, T value)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_GE(dims.size(), 1);
    CHECK_LE(dims.size(), 3);

    uint32_t remain = 3 - dims.size();
    std::vector<uint32_t> new_dims(3, 1);
    std::copy(dims.begin(), dims.end(), new_dims.begin() + remain);

    uint32_t channels = new_dims[0];
    uint32_t rows = new_dims[1];
    uint32_t cols = new_dims[2];

    arma::Cube<T> padded_data(rows, cols, channels, arma::fill::zeros);
    padded_data.fill(value);

    uint32_t min_channels = std::min(data_.n_slices, channels);
    uint32_t min_rows = std::min(data_.n_rows, rows);
    uint32_t min_cols = std::min(data_.n_cols, cols);

    for (uint32_t i = 0; i < min_channels; ++i) {
        for (uint32_t j = 0; j < min_rows; ++j) {
            for (uint32_t k = 0; k < min_cols; ++k) {
                padded_data(j, k, i) = data_(j, k, i);
            }
        }
    }

    data_ = padded_data;
    shape_ = new_dims;
}

template <typename T>
void Tensor<T>::Ones()
{
    CHECK(!data_.empty()) << "Tensor is empty";
    data_.ones();
}

template <typename T>
void Tensor<T>::Show()
{
    CHECK(!data_.empty()) << "Tensor is empty";
    std::cout << data_ << std::endl;
}

template <typename T>
void Tensor<T>::RandomNormal(T mean, T stddev)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    data_.randn();
    data_ = mean + stddev * data_;
}

template <typename T>
void Tensor<T>::RandomUniform(T low, T high)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    data_.randu();
    data_ = low + (high - low) * data_;
}

template <typename T>
void Tensor<T>::Review(const std::vector<uint32_t>& shapes)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_EQ(shape_.size(), 3) << "Tensor shape mismatch";

    const uint32_t target_channels = shapes[0];
    const uint32_t target_rows = shapes[1];
    const uint32_t target_cols = shapes[2];
    CHECK_EQ(data_.size(), target_channels * target_rows * target_cols) << "Tensor size mismatch";
    
    arma::Cube<T> target_data(target_rows, target_cols, target_channels, arma::fill::zeros);
    const uint32_t plane_size = target_rows * target_cols;

    for (uint32_t channel = 0; channel < data_.n_slices; ++channel) {
        const uint32_t plane_start = channel * data_.n_rows * data_.n_cols;
        for (uint32_t col = 0; col < data_.n_cols; ++col) {
            const T* col_ptr = data_.slice_colptr(channel, col);
            for (uint32_t row = 0; row < data_.n_rows; ++row) {
                const uint32_t pos_idx = plane_start + row * data_.n_cols + col;
                const uint32_t dst_ch = pos_idx / plane_size;
                const uint32_t dst_ch_offset = pos_idx % plane_size;
                const uint32_t dst_row = dst_ch_offset / target_cols;
                const uint32_t dst_col = dst_ch_offset % target_cols;
                target_data(dst_row, dst_col, dst_ch) = col_ptr[row];
            }
        }
    }
    data_ = std::move(target_data);
}

template <typename T>
void Tensor<T>::Reshape(const std::vector<uint32_t>& shapes, bool row_major)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    CHECK_GE(shapes.size(), 1);
    CHECK_LE(shapes.size(), 3);

    const size_t src_size = data_.size();
    const size_t dst_size = std::accumulate(shapes.begin(), shapes.end(), size_t(1), std::multiplies<size_t>());
    CHECK(src_size == dst_size);
    if (!row_major) {
        if (shapes.size() == 3) {
            data_.reshape(shapes[1], shapes[2], shapes[0]);
            shape_ = {shapes[0], shapes[1], shapes[2]};
        } else if (shapes.size() == 2) {
            data_.reshape(shapes[0], shapes[1], 1);
            shape_ = {shapes[0], shapes[1]};
        } else {
            data_.reshape(1, shapes[0], 1);
            shape_ = {shapes[0]};
        }
    } else {
        if (shapes.size() == 3) {
            this->Review({shapes[0], shapes[1], shapes[2]});
            shape_ = {shapes[0], shapes[1], shapes[2]};
        } else if (shapes.size() == 2) {
            this->Review({1, shapes[0], shapes[1]});
            shape_ = {shapes[0], shapes[1]};
        } else {
            this->Review({1, 1, shapes[0]});
            shape_ = {shapes[0]};
        }
    }
}

template <typename T>
void Tensor<T>::Transform(const std::function<T(T)>& filter)
{
    CHECK(!data_.empty()) << "Tensor is empty";
    data_.transform(filter);
}

} // namespace jeenifer