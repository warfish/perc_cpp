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

    fixedpoint16(double v) : data((int16_t)(v * (1 << kFractionBits))) {}

    operator double() const { 
        return (double)data / (1 << kFractionBits);
    }

    fixedpoint16 operator+ (fixedpoint16 rhv) {
        fixedpoint16 res;
        res.data = (data + rhv.data);
        return res;
    }

    fixedpoint16 operator* (fixedpoint16 rhv) {
        fixedpoint16 res;
        res.data = (data * rhv.data) >> fixedpoint16::kFractionBits;
        return res;
    }
};

fixedpoint16 from_double(double v) {
    return fixedpoint16(v);
}

double to_double(fixedpoint16 fp16) {
    return (double)fp16.data / (1 << fixedpoint16::kFractionBits);
}

fixedpoint16 add(fixedpoint16 lhv, fixedpoint16 rhv)
{
    return lhv + rhv;
}

fixedpoint16 mult(fixedpoint16 lhv, fixedpoint16 rhv)
{
    return lhv * rhv;
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

    // Addition
    assert_equal(
        to_double(add(from_double(0.9), from_double(0.2))),
        1.1);
    assert_equal(
        to_double(add(from_double(1.1), from_double(-0.2))),
        0.9);

    // Multiplication
    assert_equal(
        to_double(mult(from_double(1.1), from_double(0.2))),
        1.1 * 0.2);
    assert_equal(
        to_double(mult(from_double(1.1), from_double(10))),
        1.1 * 10);
}

/////////////////////////////////////////////////////////////////////////////////

//
// Non-accelerated perceptrons
//

#define DOT_SPECIALIZE 1

// 1-layer perceptron with 2 inputs
template <typename T>
struct binary_perceptron
{
    T bias;
    T weights[2];
};

// Calculate dot product for 2 inputs, 2 weights and a bias
template <typename T>
T dot(T i1, T i2, T w1, T w2, T bias)
{
    return i1 * w1 + i2 * w2 + bias;
}

#if DOT_SPECIALIZE
// dot product specialization for fixedpoints
template <>
fixedpoint16 dot<fixedpoint16>(fixedpoint16 i1, fixedpoint16 i2, fixedpoint16 w1, fixedpoint16 w2, fixedpoint16 bias)
{
    fixedpoint16 res;
    res.data = i1.data * w1.data + i2.data * w2.data + bias.data;
    return res;
}
#endif

// Activaction function
template <typename T>
static bool inline predict(binary_perceptron<T>& perc, T input1, T input2)
{
    return dot(input1, input2, perc.weights[0], perc.weights[1], perc.bias) >= 0;
}

// Train perceptron using gradient descent algorithm
template <typename T>
void train(binary_perceptron<T>& perc,
           const vector<T>& inputs1,
           const vector<T>& inputs2,
           const vector<bool>& outputs,
           unsigned nepoch,
           T rate)
{
    assert(!inputs1.empty() && !inputs2.empty() && !outputs.empty());
    assert(inputs1.size() == inputs2.size());
    assert(inputs1.size() == outputs.size());

    size_t datasize = outputs.size();
    perc.weights[0] = 0;
    perc.weights[1] = 0;
    perc.bias = 0;

    while (nepoch-- > 0) {
        for (size_t i = 0; i < datasize; ++i) {
            bool output = predict(perc, inputs1[i], inputs2[i]);
            T delta = rate * T((int)outputs[i] - (int)output);
            perc.bias = perc.bias + delta;
            perc.weights[0] = perc.weights[0] + delta * inputs1[i];
            perc.weights[1] = perc.weights[1] + delta * inputs2[i];
        }
    }
}

// Run binary perceptron in given inputs
template <typename T>
vector<bool> run(binary_perceptron<T>& perc, const vector<T>& inputs1, const vector<T>& inputs2)
{
    assert(inputs1.size() == inputs2.size());

    vector<bool> res(inputs1.size(), false);
    for (size_t i = 0; i < inputs1.size(); ++i) {
        res[i] = predict(perc, inputs1[i], inputs2[i]);
    }

    return res;
}

/////////////////////////////////////////////////////////////////////////////////

#include <sys/time.h>

template <typename T>
static void test_perceptron_builtin()
{
    //const T bias = -0.1;
    //const vector<T> weights = {0.21, -0.23};

    const vector<T> inputs1 = {
        2.78, 1.47, 3.40, 1.39, 3.06, 7.63, 5.33, 6.92, 8.68, 7.67
    };
    
    const vector<T> inputs2 = {
        2.55, 2.36, 4.40, 1.85, 3.01, 2.76, 2.09, 1.77, -0.24, 3.51
    };
    
    const vector<bool> outputs = {
        false, false, false, false, false, true, true, true, true, true
    };

    binary_perceptron<T> perc;

    struct timespec start, end;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); 
    train(perc, inputs1, inputs2, outputs, 1000000, T(0.1));
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

    unsigned long long start_ns = start.tv_sec * 1e9 + start.tv_nsec;
    unsigned long long end_ns = end.tv_sec * 1e9 + end.tv_nsec;
    unsigned long long duration_ns = end_ns - start_ns;

    printf("Train time taken (nanoseconds): %llu\n", duration_ns);
    printf("Trained weights: {%.4f, %.4f}\n", (double)perc.weights[0], (double)perc.weights[1]);
    printf("Trained bias: %.4f\n", (double)perc.bias);

    vector<bool> res = run(perc, inputs1, inputs2);
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
