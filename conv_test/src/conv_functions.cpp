#include "conv_functions.h"
#include <chrono>

using namespace std::chrono;

//-------------------------------------------------------------------------------------------------------------------------------------------------

float conv_mkl_float32(const param_struct& param, bool reorder_flag, int count_iter)
{
    high_resolution_clock::time_point _t1;
    high_resolution_clock::time_point _t2;

    auto cpu_engine = engine(engine::kind::cpu, 0);

    /* create memory for user data */
    auto conv_user_src_memory = memory({{{param.conv_src_tz}, memory::data_type::f32,
        memory::format::nchw}, cpu_engine}, param.net_src_ptr);
    auto conv_user_weights_memory = memory({{{param.conv_weights_tz},
        memory::data_type::f32, memory::format::oihw}, cpu_engine},
        param.net_weights_ptr);
    auto conv_user_bias_memory = memory({{{param.conv_bias_tz},
        memory::data_type::f32, memory::format::x}, cpu_engine},
        param.net_bias_ptr);
    auto conv_user_dst_memory = memory({{{param.conv_dst_tz}, memory::data_type::f32,
        memory::format::nchw}, cpu_engine}, param.net_dst_ptr);

    /* create memory descriptors for convolution data w/ no specified format */
    auto conv_src_md = memory::desc({param.conv_src_tz}, memory::data_type::f32,
        memory::format::any);
    auto conv_bias_md = memory::desc({param.conv_bias_tz}, memory::data_type::f32,
        memory::format::any);
    auto conv_weights_md = memory::desc({param.conv_weights_tz},
        memory::data_type::f32, memory::format::any);
    auto conv_dst_md = memory::desc({param.conv_dst_tz}, memory::data_type::f32,
        memory::format::any);

    /* create a convolution */
    auto conv_desc = convolution_forward::desc(prop_kind::forward,
        convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
        conv_dst_md, param.conv_strides, param.conv_padding, param.conv_padding,
        padding_kind::zero);
    auto conv_prim_desc =
        convolution_forward::primitive_desc(conv_desc, cpu_engine);

    std::vector<primitive> net;

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    auto conv_src_memory = conv_user_src_memory;
    if (memory::primitive_desc(conv_prim_desc.src_primitive_desc()) !=
        conv_user_src_memory.get_primitive_desc()) {
        conv_src_memory = memory(conv_prim_desc.src_primitive_desc());
        if(reorder_flag)
            net.push_back(reorder(conv_user_src_memory, conv_src_memory));
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (memory::primitive_desc(conv_prim_desc.weights_primitive_desc()) !=
        conv_user_weights_memory.get_primitive_desc()) {
        conv_weights_memory = memory(conv_prim_desc.weights_primitive_desc());
        if(reorder_flag)
            net.push_back(reorder(conv_user_weights_memory, conv_weights_memory));
    }

    auto conv_dst_memory = conv_user_dst_memory;
    if (memory::primitive_desc(conv_prim_desc.dst_primitive_desc()) !=
            conv_user_dst_memory.get_primitive_desc()) {
        conv_dst_memory = memory(conv_prim_desc.dst_primitive_desc());
    }

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv_prim_desc, conv_src_memory,
        conv_weights_memory, conv_user_bias_memory, conv_dst_memory));

    if (conv_dst_memory != conv_user_dst_memory)
    {
        if(reorder_flag)
            net.push_back(reorder(conv_dst_memory, conv_user_dst_memory));
    }

    _t1=high_resolution_clock::now();
    for(int i=0;i<count_iter;i++)
    {
        stream(stream::kind::eager).submit(net).wait();
    }
    _t2=high_resolution_clock::now();
    return float((_t2-_t1).count())/count_iter;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------

float dot_product(const param_struct& param, const float*ptr_src_rect, const float*ptr_weights_filter)
{
    int w_filter=param.conv_weights_tz[3];
    int h_filter=param.conv_weights_tz[2];
    int w_src=param.conv_src_tz[3];

    float sum=0;
    for(int j=0;j<h_filter;j++)
        for(int i=0;i<w_filter;i++)
        {
            sum+=ptr_src_rect[j*w_src+i]*ptr_weights_filter[j*w_filter+i];
        }
    return sum;
}

void conv_per_in_channel(const param_struct& param, const float*ptr_src_in_channel, const float*ptr_weights_in_channel, float*ptr_dst_out_channel)
{
    int count_h=param.conv_dst_tz[2];
    int count_w=param.conv_dst_tz[3];
    int stride_h=param.conv_strides[0];
    int stride_w=param.conv_strides[1];
    int w_src=param.conv_src_tz[3];


    for(int j=0;j<count_h;j++)
        for(int i=0;i<count_w;i++)
        {
            ptr_dst_out_channel[j*count_w+i]+=dot_product(param,ptr_src_in_channel+j*stride_h*w_src+i*stride_w,ptr_weights_in_channel);
        }
}

void conv_per_out_channel(const param_struct& param, const float*ptr_src_image, const float*ptr_weights_out_channel, float*ptr_dst_out_channel, float bias)
{
    for(int i=0;i<param.conv_dst_tz[2]*param.conv_dst_tz[3];i++)
    {
        ptr_dst_out_channel[i]=bias;
    }

    int src_in_channel_offset=param.conv_src_tz[2]*param.conv_src_tz[3];
    int weights_in_channels_offset=param.conv_weights_tz[2]*param.conv_weights_tz[3];
    for(int i=0;i<param.conv_src_tz[1];i++)
    {
        conv_per_in_channel(param,ptr_src_image+src_in_channel_offset*i,ptr_weights_out_channel+weights_in_channels_offset*i,ptr_dst_out_channel);
    }
}

void conv_per_image(const param_struct& param, const float*ptr_src_image, float*ptr_dst_image)
{
    int weights_out_channels_offset=param.conv_weights_tz[1]*param.conv_weights_tz[2]*param.conv_weights_tz[3];
    int dst_out_channel_offset=param.conv_dst_tz[2]*param.conv_dst_tz[3];
    for(int i=0;i<param.conv_weights_tz[0];i++)
    {
        conv_per_out_channel(param,ptr_src_image,param.net_weights_ptr+weights_out_channels_offset*i,ptr_dst_image+dst_out_channel_offset*i,param.net_bias_ptr[i]);
    }
}

void conv_baseline_float32(const param_struct& param)
{
    int src_offset=param.conv_src_tz[1]*param.conv_src_tz[2]*param.conv_src_tz[3];
    int dst_offset=param.conv_dst_tz[1]*param.conv_dst_tz[2]*param.conv_dst_tz[3];
    for(int i=0;i<param.batch;i++)
    {
        conv_per_image(param,param.net_src_ptr+src_offset*i,param.net_dst_ptr+dst_offset*i);
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------

void print_param(const param_struct& param)
{
    std::cout << param.name << ": ";
    printf("{%d, %d, %d, %d} (x) {%d, %d, %d, %d} -> {%d, %d, %d, %d}",
            param.conv_src_tz[0],param.conv_src_tz[1],param.conv_src_tz[2],param.conv_src_tz[3],
            param.conv_weights_tz[0],param.conv_weights_tz[1],param.conv_weights_tz[2],param.conv_weights_tz[3],
            param.conv_dst_tz[0],param.conv_dst_tz[1],param.conv_dst_tz[2],param.conv_dst_tz[3]);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------

bool verify(float*mass1,float*mass2,const param_struct& param, float time_val, int aver_count)
{
    std::cout << std::endl;
    print_param(param);
    std::cout << ", aver:" << aver_count;
    std::cout << ", time:" << lround(time_val/1000) << " mksec";

    float gflops_th=32*2.3f;
    float gflops_pr=param.batch*param.conv_dst_tz[1]*param.conv_dst_tz[2]*param.conv_dst_tz[3]*param.conv_src_tz[1]*2/time_val; //N*OC*OH*OW*IC*2

    printf(", perf:%.1f%% theor",gflops_pr/gflops_th*100);

    int count=param.conv_dst_tz[0]*param.conv_dst_tz[1]*param.conv_dst_tz[2]*param.conv_dst_tz[3];
    for(int i=0;i<count;i++)
    {
        if(mass1[i]!=mass2[i])
        {
            std::cout << ", Verify error - index:" << i << ", val1:" << mass1[i] << ", val2:" << mass2[i] << std::endl << std::endl;
            return false;
        }
    }
    std::cout << ", Verify OK" << std::endl << std::endl;
    return true;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------

void conv(param_struct& param, int aver_count)
{
    /* AlexNet: conv
     * {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
     * strides: {4, 4}
     */

    param.net_src_ptr=(float*)_mm_malloc(param.conv_src_tz[0]*param.conv_src_tz[1]*param.conv_src_tz[2]*param.conv_src_tz[3]*sizeof(float),32);
    param.net_weights_ptr=(float*)_mm_malloc(param.conv_weights_tz[0]*param.conv_weights_tz[1]*param.conv_weights_tz[2]*param.conv_weights_tz[3]*sizeof(float),32);
    param.net_bias_ptr=(float*)_mm_malloc(param.conv_bias_tz[0]*sizeof(float),32);

    float*net_dst_ptr_mkl=(float*)_mm_malloc(param.conv_dst_tz[0]*param.conv_dst_tz[1]*param.conv_dst_tz[2]*param.conv_dst_tz[3]*sizeof(float),32);
    float*net_dst_ptr_base=(float*)_mm_malloc(param.conv_dst_tz[0]*param.conv_dst_tz[1]*param.conv_dst_tz[2]*param.conv_dst_tz[3]*sizeof(float),32);


    for(int i=0;i<param.conv_src_tz[0]*param.conv_src_tz[1]*param.conv_src_tz[2]*param.conv_src_tz[3];i++)
    {
        param.net_src_ptr[i]=rand()%11-5;
    }

    for(int i=0;i<param.conv_weights_tz[0]*param.conv_weights_tz[1]*param.conv_weights_tz[2]*param.conv_weights_tz[3];i++)
    {
        param.net_weights_ptr[i]=rand()%11-5;
    }

    for(int i=0;i<param.conv_bias_tz[0];i++)
    {
        param.net_bias_ptr[i]=rand()%11-5;
    }

    const int boost_count=10;
    param.net_dst_ptr=net_dst_ptr_mkl;
    conv_mkl_float32(param,false,boost_count);
    float aver_time=conv_mkl_float32(param,false,aver_count);
    conv_mkl_float32(param,true,1);

    param.net_dst_ptr=net_dst_ptr_base;
    conv_baseline_float32(param);

    verify(net_dst_ptr_mkl,net_dst_ptr_base,param,aver_time,aver_count);

    _mm_free(param.net_src_ptr);
    _mm_free(param.net_weights_ptr);
    _mm_free(param.net_bias_ptr);
    _mm_free(net_dst_ptr_mkl);
    _mm_free(net_dst_ptr_base);

}

//-------------------------------------------------------------------------------------------------------------------------------------------------
