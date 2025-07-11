#include <gtest/gtest.h>
#include <glog/logging.h>

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("Jennifer");

    FLAGS_log_dir = "./log";
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Jtest start!\n";

    return RUN_ALL_TESTS();
}