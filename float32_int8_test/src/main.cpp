#include <iostream>
#include <fstream>
#include <chrono>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#include <sstream>
#include <stdint.h>

#define u_int8_t uint8_t
#define u_int32_t uint32_t

static int check_align(size_t align)
{
    for (size_t i = sizeof(void*); i != 0; i *= 2)
        if (align == i)
            return 0;
    return EINVAL;
}

int posix_memalign(void** ptr, size_t align, size_t size)
{
    if (check_align(align))
        return EINVAL;

    int saved_errno = errno;
    void* p = _aligned_malloc(size, align);
    if (p == NULL)
    {
        errno = saved_errno;
        return ENOMEM;
    }

    *ptr = p;
    return 0;
}

#endif

using namespace std::chrono;

enum dot_type {float32_baseline, float32_optimized_SSE128, float32_optimized_AVX256, float32_optimized_AVX256_2,
               int8_baseline, int8_optimized_SSE128, int32_baseline, int8_optimized_AVX256};

high_resolution_clock::time_point _t1;
high_resolution_clock::time_point _t2;

template <typename type_inA, typename type_inW, typename type_out>
float dot_product_baseline(type_inA*a_ptr,type_inW*w_ptr,size_t n,type_out& sum_val)
{
    type_out sum=0;
    _t1=high_resolution_clock::now();
    for(size_t i=0;i<n;i++)
    {
        sum+=a_ptr[i]*w_ptr[i];
    }
    _t2=high_resolution_clock::now();
    sum_val=sum;
    return (_t2-_t1).count();
}

float dot_product_float32_optimized_SSE128(float*a_ptr,float*w_ptr,size_t n,float& sum_val)
{
    __m128 sum4=_mm_setzero_ps();
    float*sum_ptr=(float*)(&sum4);
    __m128*a_ptr4=(__m128*)a_ptr;
    __m128*w_ptr4=(__m128*)w_ptr;
    size_t n_div4=n/4;
    _t1=high_resolution_clock::now();
    for(size_t i=0;i<n_div4;i++)
    {
        __m128 c=_mm_mul_ps(a_ptr4[i],w_ptr4[i]);
         sum4=_mm_add_ps(c,sum4);
    }
    sum_val=sum_ptr[0]+sum_ptr[1]+sum_ptr[2]+sum_ptr[3];
    _t2=high_resolution_clock::now();
    return (_t2-_t1).count();
}

float dot_product_float32_optimized_AVX256(float*a_ptr,float*w_ptr,size_t n,float& sum_val)
{
    __m256 sum8_0,sum8_1,sum8_2,sum8_3,sum8_4,sum8_5,sum8_6,sum8_7;
    sum8_0=sum8_1=sum8_2=sum8_3=sum8_4=sum8_5=sum8_6=sum8_7=_mm256_setzero_ps();
    float*sum_ptr=(float*)(&sum8_0);
    __m256*a_ptr8=(__m256*)a_ptr;
    __m256*w_ptr8=(__m256*)w_ptr;
    size_t n_div8=n/8;
    _t1=high_resolution_clock::now();
    for(size_t i=0;i<n_div8;i+=8)
    {
        sum8_0=_mm256_fmadd_ps(a_ptr8[i],w_ptr8[i],sum8_0);
        sum8_1=_mm256_fmadd_ps(a_ptr8[i+1],w_ptr8[i+1],sum8_1);
        sum8_2=_mm256_fmadd_ps(a_ptr8[i+2],w_ptr8[i+2],sum8_2);
        sum8_3=_mm256_fmadd_ps(a_ptr8[i+3],w_ptr8[i+3],sum8_3);
        sum8_4=_mm256_fmadd_ps(a_ptr8[i+4],w_ptr8[i+4],sum8_4);
        sum8_5=_mm256_fmadd_ps(a_ptr8[i+5],w_ptr8[i+5],sum8_5);
        sum8_6=_mm256_fmadd_ps(a_ptr8[i+6],w_ptr8[i+6],sum8_6);
        sum8_7=_mm256_fmadd_ps(a_ptr8[i+7],w_ptr8[i+7],sum8_7);
    }
    sum8_0=_mm256_add_ps(_mm256_add_ps(sum8_0,sum8_1),_mm256_add_ps(sum8_2,sum8_3));
    sum8_4=_mm256_add_ps(_mm256_add_ps(sum8_4,sum8_5),_mm256_add_ps(sum8_6,sum8_7));
    sum8_0=_mm256_add_ps(sum8_0,sum8_4);
    sum_val=sum_ptr[0]+sum_ptr[1]+sum_ptr[2]+sum_ptr[3]+sum_ptr[4]+sum_ptr[5]+sum_ptr[6]+sum_ptr[7];
    _t2=high_resolution_clock::now();
    return (_t2-_t1).count();
}

float dot_product_float32_optimized_AVX256_2(float*a_ptr,float*w_ptr,size_t n,float& sum_val)
{
    __m256 a0, a1, a2, a3;
    __m256 w0, w1, w2, w3;
    __m256 s0, s1, s2, s3;
    s0 = s1 = s2 = s3 =  _mm256_setzero_ps();
    size_t n_div32=n/8/4;

    _t1=high_resolution_clock::now();
    a0 = _mm256_load_ps(&a_ptr[32*0+0*8]);
    w0 = _mm256_load_ps(&w_ptr[32*0+0*8]);


    a1 = _mm256_load_ps(&a_ptr[32*0+1*8]);
    w1 = _mm256_load_ps(&w_ptr[32*0+1*8]);

    a2 = _mm256_load_ps(&a_ptr[32*0+2*8]);
    w2 = _mm256_load_ps(&w_ptr[32*0+2*8]);

    a3 = _mm256_load_ps(&a_ptr[32*0+3*8]);
    w3 = _mm256_load_ps(&w_ptr[32*0+3*8]);

    for(size_t i=0;i<n_div32-1;i++)
    {
        s0 = _mm256_fmadd_ps(a0,w0,s0);
        a0 = _mm256_load_ps(&a_ptr[32*i+32+0*8]);
        w0 = _mm256_load_ps(&w_ptr[32*i+32+0*8]);

        s1 = _mm256_fmadd_ps(a1,w1,s1);
        a1 = _mm256_load_ps(&a_ptr[32*i+32+1*8]);
        w1 = _mm256_load_ps(&w_ptr[32*i+32+1*8]);

        s2 = _mm256_fmadd_ps(a2,w2,s2);
        a2 = _mm256_load_ps(&a_ptr[32*i+32+2*8]);
        w2 = _mm256_load_ps(&w_ptr[32*i+32+2*8]);

        s3 = _mm256_fmadd_ps(a3,w3,s3);
        a3 = _mm256_load_ps(&a_ptr[32*i+32+3*8]);
        w3 = _mm256_load_ps(&w_ptr[32*i+32+3*8]);
    }
    s0 = _mm256_fmadd_ps(a0,w0,s0);
    s1 = _mm256_fmadd_ps(a1,w1,s1);
    s2 = _mm256_fmadd_ps(a2,w2,s2);
    s3 = _mm256_fmadd_ps(a3,w3,s3);

    s0 = _mm256_add_ps(s0, s1);
    s2 = _mm256_add_ps(s2, s3);
    s0 = _mm256_add_ps(s0, s2);
    float sum_ptr[8];
    _mm256_store_ps(sum_ptr, s0);
    sum_val=sum_ptr[0]+sum_ptr[1]+sum_ptr[2]+sum_ptr[3]+sum_ptr[4]+sum_ptr[5]+sum_ptr[6]+sum_ptr[7];
    _t2=high_resolution_clock::now();
    return (_t2-_t1).count();
}

float dot_product_int8_optimized_SSE128(u_int8_t* a_ptr, int8_t* w_ptr,size_t n,int32_t& sum_val)
{
    __m128i sum4=_mm_setzero_si128();
    int32_t*sum_ptr=(int32_t*)(&sum4);
    __m128i*a_ptr16=(__m128i*)a_ptr;
    __m128i*w_ptr16=(__m128i*)w_ptr;
    size_t n_div16=n/16;
    _t1=high_resolution_clock::now();
    for(size_t i=0;i<n_div16;i++)
    {
        __m128i c=_mm_maddubs_epi16(a_ptr16[i],w_ptr16[i]);
        __m128i lo=_mm_cvtepi16_epi32(c);
        __m128i hi=_mm_cvtepi16_epi32(_mm_shuffle_epi32(c,0x4e));
        sum4=_mm_add_epi32(_mm_add_epi32(lo, hi),sum4);
    }
    sum_val=sum_ptr[0]+sum_ptr[1]+sum_ptr[2]+sum_ptr[3];
    _t2=high_resolution_clock::now();
    return (_t2-_t1).count();
}

float dot_product_int8_optimized_AVX256(u_int8_t*a_ptr,int8_t*w_ptr,size_t n,int32_t& sum_val)
{
    __m256i sum16=_mm256_setzero_si256();
    int16_t*sum_ptr=(int16_t*)(&sum16);
    __m256i*a_ptr32=(__m256i*)a_ptr;
    __m256i*w_ptr32=(__m256i*)w_ptr;
    size_t n_div32=n/32;
    _t1=high_resolution_clock::now();
    for(size_t i=0;i<n_div32;i++)
    {
        __m256i c=_mm256_maddubs_epi16(a_ptr32[i],w_ptr32[i]);
        sum16=_mm256_add_epi16(c,sum16);
    }
    sum_val=sum_ptr[0]+sum_ptr[1]+sum_ptr[2]+sum_ptr[3]+sum_ptr[4]+sum_ptr[5]+sum_ptr[6]+sum_ptr[7]+
            sum_ptr[8]+sum_ptr[9]+sum_ptr[10]+sum_ptr[11]+sum_ptr[12]+sum_ptr[13]+sum_ptr[14]+sum_ptr[15];
    _t2=high_resolution_clock::now();
    return (_t2-_t1).count();
}

template <typename type_inA, typename type_inW, typename type_out>
float dot_product(size_t mem, dot_type type,type_out& sum_val)
{
    const size_t start_boost=10;
    const size_t count_iter=1000;
    const size_t kibi=1<<10;
    size_t count=mem*kibi/sizeof(type_inA);
    type_inA*massA;
    type_inW*massW;
    int err;
    err=posix_memalign((void**)&massA,32,count*sizeof(type_inA));
    err=posix_memalign((void**)&massW,32,count*sizeof(type_inW));
    if(err)
    {
        std::cout << "posix_memalign error" << std::endl;
        return -1;
    }

    srand(mem);
    for(size_t i=0;i<count;i++)
    {
        massA[i]=rand()%11;
        massW[i]=rand()%11-5;
    }

    float sum_time=0;
    for(size_t i=0; i < start_boost+count_iter; i++)
    {
        if(i==start_boost)
            sum_time=0;
        switch(type)
        {
            case float32_baseline:
            {
                sum_time+=dot_product_baseline<float,float,float>((float*)massA,(float*)massW,count,(float&)sum_val);
                break;
            }
            case float32_optimized_SSE128:
            {
                sum_time+=dot_product_float32_optimized_SSE128((float*)massA,(float*)massW,count,(float&)sum_val);
                break;
            }
            case float32_optimized_AVX256:
            {
                sum_time+=dot_product_float32_optimized_AVX256((float*)massA,(float*)massW,count,(float&)sum_val);
                break;
            }
            case float32_optimized_AVX256_2:
            {
                sum_time+=dot_product_float32_optimized_AVX256_2((float*)massA,(float*)massW,count,(float&)sum_val);
                break;
            }
            case int8_baseline:
            {
                sum_time+=dot_product_baseline<u_int8_t,int8_t,int32_t>((u_int8_t*)massA,(int8_t*)massW,count,(int32_t&)sum_val);
                break;
            }
            case int8_optimized_SSE128:
            {
                sum_time+=dot_product_int8_optimized_SSE128((u_int8_t*)massA,(int8_t*)massW,count,(int32_t&)sum_val);
                break;
            }
            case int8_optimized_AVX256:
            {
                sum_time+=dot_product_int8_optimized_AVX256((u_int8_t*)massA,(int8_t*)massW,count,(int32_t&)sum_val);
                break;
            }
            case int32_baseline:
            {
                sum_time+=dot_product_baseline<u_int32_t,int32_t,int32_t>((u_int32_t*)massA,(int32_t*)massW,count,(int32_t&)sum_val);
                break;
            }
        }
    }

#ifdef _WIN32
    _aligned_free(massA);
    _aligned_free(massW);
#else
    free(massA);
    free(massW);
#endif

    return sum_time/count_iter/count;
}

int main(int argc, char** argv)
{
    if(argc!=6)
    {
        std::cout << "Wrong argument count" << std::endl;
        std::cout << "Usage: ./float32_int8_test begin_size[Kb] end_size[Kb] step_size[Kb] mul_coef log_file" << std::endl;
        return 1;
    }

    size_t begin_size=std::stoul(std::string(argv[1]));
    size_t end_size=std::stoul(std::string(argv[2]));
    size_t step_size=std::stoul(std::string(argv[3]));
    size_t mul_coef=std::stoul(std::string(argv[4]));

    std::cout << std::endl << "params:" << std::endl;
    std::cout << "begin size: " << begin_size << "Kb" << std::endl;
    std::cout << "end size: " << end_size << "Kb" << std::endl;
    std::cout << "step size: " << step_size << "Kb" << std::endl;
    std::cout << "mul coef: " << mul_coef << std::endl << std::endl;

    std::ofstream log(argv[5]);
    float sum_val_float_base=0,sum_val_float_sse128=0,sum_val_float_avx256=0, sum_val_float_avx256_2=0;
    int32_t sum_val_int8_base=0,sum_val_int8_sse128=0,sum_val_int8_avx256=0,sum_val_int32_base=0;

    log << "memory ";
    log << "f32_base ";
    log << "f32_sse128 ";
    log << "f32_avx256 ";
    log << "i8_base ";
    log << "i8_sse128 ";
    log << "i32_base " ;
    log << "f32_avx256_2 ";
    log << "i8_avx256" << std::endl;

    for(size_t mem=begin_size;mem<=end_size;mem=mem*mul_coef+step_size)
    {
        log << mem << " ";
        log << dot_product<float,float,float>(mem,float32_baseline,sum_val_float_base) << " ";
        log << dot_product<float,float,float>(mem,float32_optimized_SSE128,sum_val_float_sse128) << " ";
        log << dot_product<float,float,float>(mem,float32_optimized_AVX256,sum_val_float_avx256) << " ";
        log << dot_product<u_int8_t,int8_t,int32_t>(mem,int8_baseline,sum_val_int8_base) << " ";
        log << dot_product<u_int8_t,int8_t,int32_t>(mem,int8_optimized_SSE128,sum_val_int8_sse128) << " ";
        log << dot_product<u_int32_t,int32_t,int32_t>(mem,int32_baseline,sum_val_int32_base) << " ";
        log << dot_product<float,float,float>(mem,float32_optimized_AVX256_2,sum_val_float_avx256_2) << " ";
        log << dot_product<u_int8_t,int8_t,int32_t>(mem,int8_optimized_AVX256,sum_val_int8_avx256) << std::endl;

        if(sum_val_float_base!=sum_val_float_sse128 || sum_val_float_base!=sum_val_float_avx256 || sum_val_float_base!=sum_val_float_avx256_2 ||
                sum_val_int8_base!=sum_val_int8_sse128 || sum_val_int8_base!=sum_val_int8_avx256 || sum_val_float_base!=sum_val_int32_base)
        {
            std::cout << "Verify: Wrong" << std::endl;
            std::cout << "Memory: " << mem << std::endl;
            std::cout << "sum_val_float_base: " << sum_val_float_base << std::endl;
            std::cout << "sum_val_float_sse128: " << sum_val_float_sse128 << std::endl;
            std::cout << "sum_val_float_avx256: " << sum_val_float_avx256 << std::endl;
            std::cout << "sum_val_float_avx256_2: " << sum_val_float_avx256_2 << std::endl;
            std::cout << "sum_val_int8_base: " << sum_val_int8_base << std::endl;
            std::cout << "sum_val_int8_sse128: " << sum_val_int8_sse128 << std::endl;
            std::cout << "sum_val_int8_avx256: " << sum_val_int8_avx256 << std::endl;
            std::cout << "sum_val_int32_base: " << sum_val_int32_base << std::endl;
            break;
        }
    }

    std::cout << "done" << std::endl << std::endl;

    return 0;
}
































