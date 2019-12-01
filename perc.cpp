#include <vector>
#include <utility>
#include <exception>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <cmath>
#include <cstdio>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////

//
// 16 bit signed fixed point 6.10
//

struct fixedpoint16 {
    enum {
        kFractionBits = 10
    };

    int16_t data;

    fixedpoint16() : data(0) {}

    explicit fixedpoint16(double v) : data((int16_t)(v * (1 << kFractionBits))) {}

    explicit operator double() const {
        return (double)data / (1 << kFractionBits);
    }
};

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
struct binary_perceptron
{
    T bias;
    T weights[2];

    // Activaction function
    bool predict(T i1, T i2) const;

    // Train perceptron using gradient descent algorithm
    void train(const vector<T>& inputs1,
               const vector<T>& inputs2,
               const vector<bool>& outputs,
               unsigned nepoch,
               T rate);

    // Run trained binary perceptron in given inputs
    vector<bool> run(const vector<T>& inputs1, const vector<T>& inputs2) const;
};

//
// Generic implementation (for T = double)
//

template <typename T>
bool binary_perceptron<T>::predict(T i1, T i2) const
{
    return (i1 * weights[0] + i2 * weights[1] + bias) >= 0;
}

template <typename T>
void binary_perceptron<T>::train(const vector<T>& inputs1,
                                 const vector<T>& inputs2,
                                  const vector<bool>& outputs,
                                unsigned nepoch,
                                T rate)
{
    assert(!inputs1.empty() && !inputs2.empty() && !outputs.empty());
    assert(inputs1.size() == inputs2.size());
    assert(inputs1.size() == outputs.size());

    weights[0] = 0;
    weights[1] = 0;
    bias = 0;

    while (nepoch-- > 0) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            bool output = predict(inputs1[i], inputs2[i]);
            T delta = rate * T((int)outputs[i] - (int)output);
            bias += delta;
            weights[0] += delta * inputs1[i];
            weights[1] += delta * inputs2[i];
        }
    }
}

template <typename T>
vector<bool> binary_perceptron<T>::run(const vector<T>& inputs1, const vector<T>& inputs2) const
{
    assert(inputs1.size() == inputs2.size());

    vector<bool> res(inputs1.size(), false);
    for (size_t i = 0; i < inputs1.size(); ++i) {
        res[i] = predict(inputs1[i], inputs2[i]);
    }

    return res;
}

//
// Specialized implementation for T = fixedpoint16
//

template <>
bool binary_perceptron<fixedpoint16>::predict(fixedpoint16 i1, fixedpoint16 i2) const
{
    return ((i1.data * weights[0].data) >> fixedpoint16::kFractionBits) +
           ((i2.data * weights[1].data) >> fixedpoint16::kFractionBits) +
           bias.data >= 0;
}

template <>
void binary_perceptron<fixedpoint16>::train(const vector<fixedpoint16>& inputs1,
                                            const vector<fixedpoint16>& inputs2,
                                            const vector<bool>& outputs,
                                            unsigned nepoch,
                                            fixedpoint16 rate)
{
    assert(!inputs1.empty() && !inputs2.empty() && !outputs.empty());
    assert(inputs1.size() == inputs2.size());
    assert(inputs1.size() == outputs.size());

    weights[0].data = 0;
    weights[1].data = 0;
    bias.data = 0;

    while (nepoch-- > 0) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            bool output = predict(inputs1[i], inputs2[i]);

            // error is either 1, 0 or -1, so no need for right shift
            int16_t diff = rate.data * ((int)outputs[i] - (int)output);
            bias.data += diff;
            weights[0].data += (diff * inputs1[i].data) >> fixedpoint16::kFractionBits;
            weights[1].data += (diff * inputs2[i].data) >> fixedpoint16::kFractionBits;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////

#include <sys/time.h>

template <typename T>
static void test_perceptron_builtin()
{
    //const T bias = -0.1;
    //const vector<T> weights = {0.21, -0.23};

    const vector<T> inputs1 = {
        T(2.78), T(1.47), T(3.40), T(1.39), T(3.06), T(7.63), T(5.33), T(6.92), T(8.68), T(7.67)
    };
    
    const vector<T> inputs2 = {
        T(2.55), T(2.36), T(4.40), T(1.85), T(3.01), T(2.76), T(2.09), T(1.77), T(-0.24), T(3.51)
    };
    
    const vector<bool> outputs = {
        false, false, false, false, false, true, true, true, true, true
    };

    binary_perceptron<T> perc;

    struct timespec start, end;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); 
    perc.train(inputs1, inputs2, outputs, 1000000, T(0.1));
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

    unsigned long long start_ns = start.tv_sec * 1e9 + start.tv_nsec;
    unsigned long long end_ns = end.tv_sec * 1e9 + end.tv_nsec;
    unsigned long long duration_ns = end_ns - start_ns;

    printf("Train time taken (nanoseconds): %llu\n", duration_ns);
    printf("Trained weights: {%.4f, %.4f}\n", (double)perc.weights[0], (double)perc.weights[1]);
    printf("Trained bias: %.4f\n", (double)perc.bias);

    vector<bool> res = perc.run(inputs1, inputs2);
    for (size_t i = 0; i < outputs.size(); ++i) {
        assert(res[i] == outputs[i]);
    }
}

int main()
{
    test_fixedpoint16();

    cout << endl << "binary_perceptron<double>" << endl;
    test_perceptron_builtin<double>();

    cout << endl << "binary_perceptron<fixedpoint16>" << endl;
    test_perceptron_builtin<fixedpoint16>();

    return 0;
}
