#include <vector>
#include <utility>
#include <exception>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace std;

////////////////////////////////////////////////////////////////////////////////

//
// Simple perceptron based on floating point calculations
//

struct perceptron_flt
{
    typedef double data_type;

    double bias;
    vector<double> weights;

    // Activaction function
    bool predict(const vector<double>& inputs) const;

    // Train perceptron using gradient descent algorithm
    void train(const vector<vector<double>>& rows,
               const vector<bool>& outputs,
               size_t ninputs,
               unsigned nepoch,
               float rate);
};

bool perceptron_flt::predict(const vector<double>& inputs) const
{
    double acc = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        acc += inputs[i] * weights[i];
    }

    return acc >= 0;
}

void perceptron_flt::train(const vector<vector<double>>& rows,
                           const vector<bool>& outputs,
                           size_t ninputs,
                           unsigned nepoch,
                           float rate)
{
    size_t nrows = rows.size();
    weights = vector<double>(ninputs, 0);
    bias = 0;

    while (nepoch-- > 0) {
        for (size_t i = 0; i < nrows; ++i) {
            const vector<double>& inputs = rows[i];
            bool output = predict(inputs);

            int error = (int)outputs[i] - (int)output;
            double delta = rate * error;

            bias += delta;
            for (size_t w = 0; w < weights.size(); ++w) {
                weights[w] += delta * inputs[w];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// Simple perceptron based on integer calculations
//

// 16 bit signed fixed point 8.8
struct fixedpoint16 {
    enum {
        kFractionBits = 8
    };

    int16_t data;

    fixedpoint16() : data(0) {}

    explicit fixedpoint16(double v) : data((int16_t)(v * (1 << kFractionBits))) {}

    explicit operator double() const {
        return (double)data / (1 << kFractionBits);
    }
};

static bool fuzzy_compare(double a, double b, double tolerance = 0.005)
{
    return fabs(a - b) <= tolerance;
}

static void assert_equal(double a, double b)
{
    if (!fuzzy_compare(a, b)) {
        printf("%.10f != %.10f\n", a, b);
        assert(0);
    }
}

static void assert_equal_fixedpoint(double val)
{
    // Convert to fixedpoint and back to double
    assert_equal((double)(fixedpoint16)(val), val);
}

static void test_fixedpoint16()
{
    assert(fuzzy_compare(0.009, 0.01));
    assert(fuzzy_compare(-0.002, 0.003));

    assert_equal_fixedpoint(0.0);
    assert_equal_fixedpoint(1.0);
    assert_equal_fixedpoint(1.1);
    assert_equal_fixedpoint(-1.1);
    
    // We care about 2 digits of fraction precision
    for (double i = .01; i < 1.0; i += .01) {
        assert_equal_fixedpoint(i);
    }
}

#include <immintrin.h>
#include <string.h>

#define AVX512_ALIGN            alignas(64)
#define AVX512_TOTAL_INT16      (64 / sizeof(int16_t))
#define AVX512_TOTAL_INT32      (64 / sizeof(int32_t))

static bool check_vnni_cpuid()
{
    uint32_t eax, ebx, ecx, edx;
    asm volatile ("cpuid"
                  : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
                  : "a" (0x7), "c" (0x0));

    return (ecx & (1 << 11)) != 0;
}

// Check for AVX512 VNNI instruction set
static inline bool is_vnni_supported()
{
    static bool is_supported = check_vnni_cpuid();
    return is_supported;
}

// Calculate dot product using VNNI instructions
static int32_t vnni_dot(const fixedpoint16* a,
                        const fixedpoint16* b,
                        size_t nelem)
{

    size_t nchunks = nelem / AVX512_TOTAL_INT16;

    // Intermidiate summs
    AVX512_ALIGN int32_t sums[AVX512_TOTAL_INT32] = {0};

    // For each full chunk use entire AVX register
    for (size_t i = 0; i < nchunks; ++i) {
        asm volatile (
            "vmovdqu64 %0, %%zmm0 \r\n"
            "vmovdqu64 %1, %%zmm1 \r\n"
            "vmovdqa64 %2, %%zmm2 \r\n"
            "vpdpwssd %%zmm0, %%zmm1, %%zmm2 \r\n"
            "vmovdqa32 %%zmm2, %2 \r\n"
            : : "m"(*a), "m"(*b), "m"(sums) : "zmm0", "zmm1", "zmm2");

        a += AVX512_TOTAL_INT16;
        b += AVX512_TOTAL_INT16;
        nelem -= AVX512_TOTAL_INT16;
    }

    // Handle remainder, if any
    if (nelem > 0) {
        AVX512_ALIGN int16_t tmpa[AVX512_TOTAL_INT16] = {0};
        AVX512_ALIGN int16_t tmpb[AVX512_TOTAL_INT16] = {0};

        memcpy(tmpa, a, nelem * sizeof(*a));
        memcpy(tmpb, b, nelem * sizeof(*b));

        asm volatile(
            "vmovdqa64 %0, %%zmm0 \r\n"
            "vmovdqa64 %1, %%zmm1 \r\n"
            "vmovdqa64 %2, %%zmm2 \r\n"
            "vpdpwssd %%zmm0, %%zmm1, %%zmm2 \r\n"
            "vmovdqa32 %%zmm2, %2 \r\n"
            : : "m"(tmpa), "m"(tmpb), "m"(sums) : "zmm0", "zmm1", "zmm2");
    }

    // Reduce AVX register to a sum
    // There exists an _mm512_reduce_add_epi32 intrinsic but it does not appear to be faster
    int32_t acc = 0;
    for (size_t i = 0; i < AVX512_TOTAL_INT32; ++i) {
        acc += sums[i] >> fixedpoint16::kFractionBits;
    }

    return acc;
}

// Software implementation of a dot product
static int32_t sw_dot(const fixedpoint16* a,
                      const fixedpoint16* b,
                      size_t nelem)
{
    int32_t acc = 0;
    for (size_t i = 0; i < nelem; ++i) {
        acc += ((int32_t)a[i].data * b[i].data) >> fixedpoint16::kFractionBits;
    }

    return acc;
}

static void test_dot()
{
    enum {
        kElements = 32 + 32 + 2, // 2 full AVX512 regs + 2 extra elements
    };

    fixedpoint16 a[kElements];
    fixedpoint16 b[kElements];

    for (size_t i = 0; i < kElements; ++i) {
        a[i] = (fixedpoint16)i;
        b[i] = (fixedpoint16)(i + 10);
    }

    int32_t vnni = vnni_dot(a, b, kElements);
    int32_t sw = sw_dot(a, b, kElements);

    if (vnni != sw) {
        printf("sw dot = %d, vnni dot = %d\n", sw, vnni);
        assert(vnni == sw);
    }
}

struct perceptron_int
{
    typedef fixedpoint16 data_type;

    fixedpoint16 bias;
    vector<fixedpoint16> weights;

    bool vnni_supported;
    bool vnni_enabled;
    int32_t(*dot_func)(const fixedpoint16* a, const fixedpoint16* b, size_t nelem);

    explicit perceptron_int(bool enable_vnni) :
        vnni_supported(is_vnni_supported()),
        vnni_enabled(vnni_supported ? enable_vnni : false),
        dot_func(vnni_enabled ? vnni_dot : sw_dot)
    {
    }

    // Activaction function
    bool predict(const vector<fixedpoint16>& inputs) const;

    // Train perceptron using gradient descent algorithm
    void train(const vector<vector<fixedpoint16>>& rows,
               const vector<bool>& outputs,
               size_t ninputs,
               unsigned nepoch,
               float rate);
};

bool perceptron_int::predict(const vector<fixedpoint16>& inputs) const
{
    int32_t acc = this->bias.data;
    acc += this->dot_func(inputs.data(), weights.data(), inputs.size());
    return acc >= 0;
}

void perceptron_int::train(const vector<vector<fixedpoint16>>& rows,
                           const vector<bool>& outputs,
                           size_t ninputs,
                           unsigned nepoch,
                           float rate)
{
    fixedpoint16 fp_rate(rate);
    size_t nrows = rows.size();

    weights = vector<fixedpoint16>(ninputs);
    bias = fixedpoint16(0);

    while (nepoch-- > 0) {
        for (size_t i = 0; i < nrows; ++i) {
            const vector<fixedpoint16>& inputs = rows[i];
            bool output = predict(inputs);

            int error = (int)outputs[i] - (int)output;

            // error is either 1, 0 or -1, so no need for right shift
            int16_t delta = fp_rate.data * error;
            bias.data += delta;

            for (size_t w = 0; w < weights.size(); ++w) {
                weights[w].data += (delta * inputs[w].data) >> fixedpoint16::kFractionBits;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////

#include <sys/time.h>
#include "sonar.h"

template <typename perc_type>
static void test_perceptron_builtin(perc_type& perc)
{
    typedef typename perc_type::data_type data_type;

    // Load sonar dataset
    vector<vector<data_type>> rows(SONAR_DATASET_ROWS);
    vector<bool> outputs(SONAR_DATASET_ROWS);

    for (size_t i = 0; i < SONAR_DATASET_ROWS; ++i) {
        vector<data_type> row(SONAR_DATASET_INPUTS);
        for (size_t j = 0; j < SONAR_DATASET_INPUTS; ++j) {
            row[j] = (data_type)g_sonar_dataset[i][j];
        }

        rows[i] = row;
        outputs[i] = g_sonar_dataset[i][SONAR_DATASET_INPUTS] != 0.0;
    }

    // Run weight training
    struct timespec start, end;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); 
    perc.train(rows, outputs, SONAR_DATASET_INPUTS, 100000, 0.1);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

    unsigned long long start_ns = start.tv_sec * 1e9 + start.tv_nsec;
    unsigned long long end_ns = end.tv_sec * 1e9 + end.tv_nsec;
    unsigned long long duration_ns = end_ns - start_ns;

    printf("Train time taken (nanoseconds): %llu\n", duration_ns);

    /*
    printf("Trained weights: ");
    for (auto w : perc.weights) {
        printf("%.4f ", (double)w);
    }
    printf("\n");
    printf("Trained bias: %.4f\n", (double)perc.bias);
    */

    // Calculate resulting accuracy
    vector<bool> res(rows.size(), false);
    for (size_t i = 0; i < rows.size(); ++i) {
        res[i] = perc.predict(rows[i]);
    }

    size_t correct = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (res[i] == outputs[i]) {
            correct++;
        }
    }

    printf("Accuracy %zu/%d (%.2f%%)\n", correct, SONAR_DATASET_ROWS, (float)correct / SONAR_DATASET_ROWS * 100);
}

int main()
{
    test_fixedpoint16();

    printf("CPU supports AVX512 VNNI: %s\n", is_vnni_supported() ? "yes" : "no");
    if (is_vnni_supported()) {
        test_dot();
    }

    {
        printf("\nFloat point-based:\n");

        perceptron_flt perc;
        test_perceptron_builtin(perc);
    }

    {
        printf("\nFixed point-based (no VNNI):\n");

        perceptron_int perc(false);
        test_perceptron_builtin(perc);
    }

    {
        printf("\nFixed point-based (VNNI):\n");

        perceptron_int perc(true);
        test_perceptron_builtin(perc);
    }

    return 0;
}
