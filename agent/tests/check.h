#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace check {

struct TestCase {
    std::string name;
    std::function<void()> body;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline int& failures() {
    static int count = 0;
    return count;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

inline void fail(const char* file, int line, const std::string& message) {
    ++failures();
    std::cerr << file << ":" << line << ": " << message << "\n";
}

template <typename A, typename B>
void expect_equal(const A& actual, const B& expected, const char* expression, const char* file, int line) {
    if (actual == expected) return;
    std::ostringstream message;
    message << expression << " -> got " << actual << ", expected " << expected;
    fail(file, line, message.str());
}

inline void expect_near(double actual, double expected, double tolerance, const char* expression,
                        const char* file, int line) {
    if (std::fabs(actual - expected) <= tolerance) return;
    std::ostringstream message;
    message << expression << " -> got " << actual << ", expected " << expected << " +/- " << tolerance;
    fail(file, line, message.str());
}

inline int run_all() {
    int failed_tests = 0;
    for (const TestCase& test : registry()) {
        int before = failures();
        test.body();
        bool passed = failures() == before;
        if (!passed) ++failed_tests;
        std::cout << (passed ? "[ PASS ] " : "[ FAIL ] ") << test.name << "\n";
    }
    std::cout << registry().size() - failed_tests << "/" << registry().size() << " tests passed\n";
    return failed_tests == 0 ? 0 : 1;
}

}

#define CHECK_CONCAT_INNER(a, b) a##b
#define CHECK_CONCAT(a, b) CHECK_CONCAT_INNER(a, b)
#define TEST_CASE(name)                                                         \
    static void CHECK_CONCAT(test_body_, __LINE__)();                           \
    static check::Registrar CHECK_CONCAT(test_registrar_, __LINE__)(name,       \
                                                                    CHECK_CONCAT(test_body_, __LINE__)); \
    static void CHECK_CONCAT(test_body_, __LINE__)()

#define CHECK(condition) \
    do { if (!(condition)) check::fail(__FILE__, __LINE__, "CHECK(" #condition ") failed"); } while (0)
#define REQUIRE(condition) \
    do { if (!(condition)) { check::fail(__FILE__, __LINE__, "REQUIRE(" #condition ") failed"); return; } } while (0)
#define CHECK_EQ(actual, expected) check::expect_equal((actual), (expected), #actual, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tolerance) \
    check::expect_near((actual), (expected), (tolerance), #actual, __FILE__, __LINE__)
