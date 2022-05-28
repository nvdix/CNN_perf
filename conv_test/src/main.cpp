#include "conv_functions.h"

int main()
{
    param_struct param;

    //AlexNet conv
    param.batch = 8;
    param.conv_src_tz = {param.batch, 3, 227, 227};
    param.conv_weights_tz = {96, 3, 11, 11};
    param.conv_bias_tz = {96};
    param.conv_dst_tz = {param.batch, 96, 55, 55};
    param.conv_strides = {4, 4};
    param.conv_padding = {0, 0};
    param.name="AlexNet conv";
    conv(param,100);

    //base conv
    param.batch = 1;
    param.conv_src_tz = {param.batch, 256, 28, 28};
    param.conv_weights_tz = {64, 256, 1, 1};
    param.conv_bias_tz = {64};
    param.conv_dst_tz = {param.batch, 64, 28, 28};
    param.conv_strides = {1, 1};
    param.conv_padding = {0, 0};
    param.name="base conv";
    conv(param,10000);

    //googlenet_v2 conv1
    param.batch = 1;
    param.conv_src_tz = {1,64,56,56};
    param.conv_weights_tz = {64,64,1,1};
    param.conv_bias_tz = {64};
    param.conv_dst_tz = {1,64,56,56};
    param.conv_strides = {1, 1};
    param.conv_padding = {0, 0};
    param.name="googlenet_v2 conv1";
    conv(param,10000);

    //googlenet_v2 conv2
    param.batch = 1;
    param.conv_src_tz = {1,192,28,28};
    param.conv_weights_tz = {32,192,1,1};
    param.conv_bias_tz = {32};
    param.conv_dst_tz = {1,32,28,28};
    param.conv_strides = {1, 1};
    param.conv_padding = {0, 0};
    param.name="googlenet_v2 conv2";
    conv(param,10000);

    //googlenet_v2 conv3
    param.batch = 1;
    param.conv_src_tz = {1,320,28,28};
    param.conv_weights_tz = {64,320,1,1};
    param.conv_bias_tz = {64};
    param.conv_dst_tz = {1,64,28,28};
    param.conv_strides = {1, 1};
    param.conv_padding = {0, 0};
    param.name="googlenet_v2 conv3";
    conv(param,100);

    //googlenet_v2 conv4
    param.batch = 1;
    param.conv_src_tz = {1,576,14,14};
    param.conv_weights_tz = {96,576,1,1};
    param.conv_bias_tz = {96};
    param.conv_dst_tz = {1,96,14,14};
    param.conv_strides = {1, 1};
    param.conv_padding = {0, 0};
    param.name="googlenet_v2 conv4";
    conv(param,1000);

    //googlenet_v2 conv5
    param.batch = 1;
    param.conv_src_tz = {1,1024,7,7};
    param.conv_weights_tz = {192,1024,1,1};
    param.conv_bias_tz = {192};
    param.conv_dst_tz = {1,192,7,7};
    param.conv_strides = {1, 1};
    param.conv_padding = {0, 0};
    param.name="googlenet_v2 conv5";
    conv(param,1000);

    std::cout << "done test!" << std::endl << std::endl;
    return 0;
}















































