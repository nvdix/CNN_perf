#ifndef CONV_FUNCTIONS_H
#define CONV_FUNCTIONS_H

#include <iostream>
#include <numeric>
#include <string>
//#include <mkldnn.hpp>
#include "dnnl.hpp"
#include <malloc.h>

//using namespace mkldnn;
using namespace dnnl;

struct param_struct
{
    int batch;
    memory::dims conv_src_tz;
    memory::dims conv_weights_tz;
    memory::dims conv_bias_tz;
    memory::dims conv_dst_tz;
    memory::dims conv_strides;
    memory::dims conv_padding;

    float*net_src_ptr;
    float*net_dst_ptr;
    float*net_weights_ptr;
    float*net_bias_ptr;

    std::string name;
};

void conv(param_struct& param, int aver_count);

#endif // CONV_FUNCTIONS_H
