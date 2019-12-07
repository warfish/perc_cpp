#include <vector>
#include <utility>
#include <exception>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////

//
// 16 bit signed fixed point 8.8
//

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

    fixedpoint16 operator + (const fixedpoint16& other) const {
        fixedpoint16 res;
        res.data = (data + other.data);
        return res;
    }

    fixedpoint16& operator += (const fixedpoint16& other) {
        data += other.data;
        return *this;
    }

    fixedpoint16 operator * (const fixedpoint16& other) const {
        fixedpoint16 res;
        res.data = (data * other.data) >> kFractionBits;
        return res;
    }

    fixedpoint16& operator *= (const fixedpoint16& other) {
        data = (data * other.data) >> kFractionBits;
        return *this;
    }

    bool operator >= (int val) const {
        return data >= val;
    }
};

fixedpoint16 from_int(int v)
{
    fixedpoint16 res;
    res.data = (int16_t)(v * (1 << fixedpoint16::kFractionBits));
    return res;
}

fixedpoint16 from_double(double v)
{
    fixedpoint16 res;
    res.data = (int16_t)(v * (1 << fixedpoint16::kFractionBits));
    return res;
}

double to_double(fixedpoint16 fp16) {
    return (double)fp16.data / (1 << fixedpoint16::kFractionBits);
}

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
    assert_equal(to_double(from_double(val)), val);
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

/////////////////////////////////////////////////////////////////////////////////

//
// Non-accelerated perceptrons
//

// 1-layer perceptron with 2 inputs
template <typename T>
struct perceptron
{
    T bias;
    vector<T> weights;

    // Helper to calculate dot product for inputs and weights
    T dot(const vector<T>& inputs) const;

    // Activaction function
    bool predict(const vector<T>& inputs) const;

    // Train perceptron using gradient descent algorithm
    void train(const vector<vector<T>>& rows,
               const vector<bool>& outputs,
               size_t ninputs,
               unsigned nepoch,
               T rate);

    // Run trained binary perceptron in given inputs
    vector<bool> run(const vector<vector<T>>& rows) const;

    struct dataset
    {
        vector<vector<T>> inputs;
        vector<bool> outputs;
    };
};

//
// Generic implementation (for T = double)
//

template <typename T>
T perceptron<T>::dot(const vector<T>& inputs) const
{
    T acc = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        acc += inputs[i] * weights[i];
    }

    return acc;
}

template <typename T>
bool perceptron<T>::predict(const vector<T>& inputs) const
{
    assert(inputs.size() == weights.size());
    return dot(inputs) >= 0;
}

template <typename T>
void perceptron<T>::train(const vector<vector<T>>& rows,
                          const vector<bool>& outputs,
                          size_t ninputs,
                          unsigned nepoch,
                          T rate)
{
    assert(!rows.empty());
    assert(ninputs != 0);
    assert(rows.size() == outputs.size());

    size_t nrows = rows.size();
    weights = vector<T>(ninputs, 0);
    bias = 0;

    while (nepoch-- > 0) {
        for (size_t i = 0; i < nrows; ++i) {
            const vector<T>& inputs = rows[i];
            assert(inputs.size() == ninputs);

            bool output = predict(inputs);
            int error = (int)outputs[i] - (int)output;
            T delta = rate * error;

            bias += delta;
            for (size_t w = 0; w < weights.size(); ++w) {
                weights[w] += delta * inputs[w];
            }
        }
    }
}

template <typename T>
vector<bool> perceptron<T>::run(const vector<vector<T>>& rows) const
{
    assert (!rows.empty());

    vector<bool> res(rows.size(), false);
    for (size_t i = 0; i < rows.size(); ++i) {
        res[i] = predict(rows[i]);
    }

    return res;
}

//
// Specialized implementation for T = fixedpoint16
//

#include <immintrin.h>
#include <string.h>

#define AVX512_ALIGN            alignas(64)
#define AVX512_TOTAL_INT16      (64 / sizeof(int16_t))
#define AVX512_TOTAL_INT32      (64 / sizeof(int32_t))

static bool g_vnni_enabled = false;

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
    int32_t acc = 0;

    // For each full chunk use entire AVX register
    for (size_t i = 0; i < nchunks; ++i) {
        __m512i areg = _mm512_loadu_si512(a);
        __m512i breg = _mm512_loadu_si512(b);
        __m512i srcreg = _mm512_load_si512(sums);

        __m512i dstreg = _mm512_dpwssd_epi32(srcreg, areg, breg);
        _mm512_store_si512(sums, dstreg);

        a += AVX512_TOTAL_INT16;
        b += AVX512_TOTAL_INT16;
        nelem -= AVX512_TOTAL_INT16;
    }

    // Handle remainder, if any
    if (nelem > 0) {
        AVX512_ALIGN int16_t tmp[AVX512_TOTAL_INT16] = {0};

        memcpy(tmp, a, nelem * sizeof(*a));
        __m512i areg = _mm512_load_si512(tmp);

        memcpy(tmp, b, nelem * sizeof(*b));
        __m512i breg = _mm512_load_si512(tmp);

        __m512i srcreg = _mm512_load_si512(sums);

        __m512i dstreg = _mm512_dpwssd_epi32(srcreg, areg, breg);
        _mm512_store_si512(sums, dstreg);
    }

    // Combine intermidiate sums
    // TODO: AVX instruction?
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
        a[i] = from_int(i);
        b[i] = from_int(i + 10);
    }

    int32_t vnni = vnni_dot(a, b, kElements);
    int32_t sw = sw_dot(a, b, kElements);

    printf("sw dot = %d, vnni dot = %d\n", sw, vnni);
    assert(vnni == sw);
}

template <>
bool perceptron<fixedpoint16>::predict(const vector<fixedpoint16>& inputs) const
{
    int32_t acc = bias.data;
    if (g_vnni_enabled && is_vnni_supported()) {
        acc += vnni_dot(inputs.data(), weights.data(), inputs.size());
    } else {
        acc += sw_dot(inputs.data(), weights.data(), inputs.size());
    }

    return acc >= 0;
}

template <>
void perceptron<fixedpoint16>::train(const vector<vector<fixedpoint16>>& rows,
                                     const vector<bool>& outputs,
                                     size_t ninputs,
                                     unsigned nepoch,
                                     fixedpoint16 rate)
{
    assert(!rows.empty());
    assert(ninputs != 0);
    assert(rows.size() == outputs.size());

    size_t nrows = rows.size();
    weights = vector<fixedpoint16>(ninputs);
    bias.data = 0;

    while (nepoch-- > 0) {
        for (size_t i = 0; i < nrows; ++i) {
            const vector<fixedpoint16>& inputs = rows[i];
            assert(inputs.size() == ninputs);

            bool output = predict(inputs);
            int error = (int)outputs[i] - (int)output;

            // error is either 1, 0 or -1, so no need for right shift
            int16_t delta = rate.data * error;
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

template <typename T>
static void test_perceptron_builtin()
{
    // Load sonar dataset
    vector<vector<T>> rows(SONAR_DATASET_ROWS);
    vector<bool> outputs(SONAR_DATASET_ROWS);

    for (size_t i = 0; i < SONAR_DATASET_ROWS; ++i) {
        vector<T> row(SONAR_DATASET_INPUTS);
        for (size_t j = 0; j < SONAR_DATASET_INPUTS; ++j) {
            row[j] = T(g_sonar_dataset[i][j]);
        }

        rows[i] = row;
        outputs[i] = g_sonar_dataset[i][SONAR_DATASET_INPUTS] != 0.0;
    }

    // Run weight training
    perceptron<T> perc;
    struct timespec start, end;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); 
    perc.train(rows, outputs, SONAR_DATASET_INPUTS, 10000, T(0.1));
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

    unsigned long long start_ns = start.tv_sec * 1e9 + start.tv_nsec;
    unsigned long long end_ns = end.tv_sec * 1e9 + end.tv_nsec;
    unsigned long long duration_ns = end_ns - start_ns;

    printf("Train time taken (nanoseconds): %llu\n", duration_ns);
    printf("Trained weights: ");
    for (auto w : perc.weights) {
        printf("%.4f ", (double)w);
    }
    printf("\n");
    printf("Trained bias: %.4f\n", (double)perc.bias);

    // Calculate resulting accuracy
    vector<bool> res = perc.run(rows);
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

    printf("\nbinary_perceptron<double>\n");
    test_perceptron_builtin<double>();

    printf("\nbinary_perceptron<fixedpoint16> (no VNNI)\n");
    g_vnni_enabled = false;
    test_perceptron_builtin<fixedpoint16>();

    printf("\nbinary_perceptron<fixedpoint16> (VNNI)\n");
    g_vnni_enabled = true;
    test_perceptron_builtin<fixedpoint16>();

    return 0;
}
