#include <gtest/gtest.h>

#include "jennifer/tensor.hpp"

template <typename T>
class TensorTest : public ::testing::Test {
protected:
};

// using TestTypes = ::testing::Types<float, int32_t, uint8_t>;
using TestTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(TensorTest, TestTypes);

namespace jennifer
{

TYPED_TEST(TensorTest, tensor_init1)
{
    Tensor<TypeParam> f1(3, 224, 224);
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
    ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TYPED_TEST(TensorTest, tensor_init1_1d)
{
    Tensor<TypeParam> f1(3);
    const auto& raw_shapes = f1.raw_shape();
    ASSERT_EQ(raw_shapes.size(), 1);
    ASSERT_EQ(raw_shapes.at(0), 3);
}

TYPED_TEST(TensorTest, tensor_init1_2d)
{
    Tensor<TypeParam> f1(32, 24);
    const auto& raw_shapes = f1.raw_shape();
    ASSERT_EQ(raw_shapes.size(), 2);
    ASSERT_EQ(raw_shapes.at(0), 32);
    ASSERT_EQ(raw_shapes.at(1), 24);
}

TYPED_TEST(TensorTest, test_init1_2d_1)
{
    Tensor<TypeParam> f1(1, 24);
    const auto& raw_shapes = f1.raw_shape();
    ASSERT_EQ(raw_shapes.size(), 1);
    ASSERT_EQ(raw_shapes.at(0), 24);
}

TYPED_TEST(TensorTest, tensor_init2)
{
    Tensor<TypeParam> f1(std::vector<uint32_t>{3, 224, 224});
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
    ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TYPED_TEST(TensorTest, tensor_init3)
{
    Tensor<TypeParam> f1(std::vector<uint32_t>{1, 13, 14});
    ASSERT_EQ(f1.channels(), 1);
    ASSERT_EQ(f1.rows(), 13);
    ASSERT_EQ(f1.cols(), 14);
    ASSERT_EQ(f1.size(), 13 * 14);
}

TYPED_TEST(TensorTest, tensor_init4)
{
    Tensor<TypeParam> f1(std::vector<uint32_t>{13, 15});
    ASSERT_EQ(f1.channels(), 1);
    ASSERT_EQ(f1.rows(), 13);
    ASSERT_EQ(f1.cols(), 15);
    ASSERT_EQ(f1.size(), 13 * 15);
}

TYPED_TEST(TensorTest, tensor_init5)
{
    Tensor<TypeParam> f1(std::vector<uint32_t>{16, 13, 15});
    ASSERT_EQ(f1.channels(), 16);
    ASSERT_EQ(f1.rows(), 13);
    ASSERT_EQ(f1.cols(), 15);
    ASSERT_EQ(f1.size(), 16 * 13 * 15);
}

TYPED_TEST(TensorTest, copy_construct1)
{
    Tensor<TypeParam> f1(3, 224, 224);
    f1.RandomNormal();
    Tensor<TypeParam> f2(f1);
    ASSERT_EQ(f2.channels(), 3);
    ASSERT_EQ(f2.rows(), 224);
    ASSERT_EQ(f2.cols(), 224);
    ASSERT_TRUE(arma::approx_equal(f2.get_data(), f1.get_data(), "absdiff", 1e-4));
}

TYPED_TEST(TensorTest, copy_construct2)
{
    Tensor<TypeParam> f1(3, 2, 1);
    Tensor<TypeParam> f2(3, 224, 224);
    f2.RandomNormal();
    f1 = f2;
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
    ASSERT_TRUE(arma::approx_equal(f2.get_data(), f1.get_data(), "absdiff", 1e-4));
}

TYPED_TEST(TensorTest, move_construct1)
{
    Tensor<TypeParam> f1(3, 2, 1);
    Tensor<TypeParam> f2(3, 224, 224);
    f1 = std::move(f2);
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
    ASSERT_EQ(f2.get_data().memptr(), nullptr);
}

TYPED_TEST(TensorTest, move_construct2)
{
    Tensor<TypeParam> f2(3, 224, 224);
    Tensor<TypeParam> f1(std::move(f2));
    ASSERT_EQ(f1.channels(), 3);
    ASSERT_EQ(f1.rows(), 224);
    ASSERT_EQ(f1.cols(), 224);
    ASSERT_EQ(f2.get_data().memptr(), nullptr);
}

TYPED_TEST(TensorTest, set_data)
{
    Tensor<TypeParam> f2(3, 224, 224);
    arma::Cube<TypeParam> cube1(224, 224, 3);
    cube1.randn();
    f2.set_data(cube1);
    ASSERT_TRUE(arma::approx_equal(f2.get_data(), cube1, "absdiff", 1e-4));
}

TYPED_TEST(TensorTest, fill1)
{
    Tensor<TypeParam> f3(3, 3, 3);
    std::vector<TypeParam> values;
    for (int i = 0; i < 27; ++i) {
        values.push_back(static_cast<TypeParam>(i));
    }
    f3.Fill(values);
    int index = 0;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < f3.rows(); ++i) {
            for (int j = 0; j < f3.cols(); ++j) {
                ASSERT_EQ(f3.at(c, i, j), values[index]);
                index += 1;
            }
        }
    }
}

TYPED_TEST(TensorTest, flatten1)
{
    Tensor<TypeParam> f3(3, 3, 3);
    std::vector<TypeParam> values;
    for (int i = 0; i < 27; ++i) {
        values.push_back(static_cast<TypeParam>(i));
    }
    f3.Fill(values);
    f3.Flatten(false);
    ASSERT_EQ(f3.channels(), 1);
    ASSERT_EQ(f3.rows(), 1);
    ASSERT_EQ(f3.cols(), 27);
    
    ASSERT_EQ(f3.index(0), 0);
    ASSERT_EQ(f3.index(1), 3);
    ASSERT_EQ(f3.index(2), 6);

    ASSERT_EQ(f3.index(3), 1);
    ASSERT_EQ(f3.index(4), 4);
    ASSERT_EQ(f3.index(5), 7);

    ASSERT_EQ(f3.index(6), 2);
    ASSERT_EQ(f3.index(7), 5);
    ASSERT_EQ(f3.index(8), 8);
}

} // namespace jennifer
