#include <limits>

#include "check.h"
#include "json_writer.h"
#include "records.h"

using namespace mlplat;

TEST_CASE("escape_json escapes quotes, backslashes and control characters") {
    CHECK_EQ(escape_json("a\"b\\c"), std::string("\"a\\\"b\\\\c\""));
    CHECK_EQ(escape_json("line\nnext\ttab"), std::string("\"line\\nnext\\ttab\""));
    CHECK_EQ(escape_json(std::string_view("\x01", 1)), std::string("\"\\u0001\""));
}

TEST_CASE("escape_json keeps valid UTF-8 and replaces invalid bytes") {
    CHECK_EQ(escape_json("caf\xC3\xA9"), std::string("\"caf\xC3\xA9\""));
    CHECK_EQ(escape_json("bad\xFFz"), std::string("\"bad\\ufffdz\""));
    CHECK_EQ(escape_json("cut\xC3"), std::string("\"cut\\ufffd\""));
}

TEST_CASE("JsonWriter builds nested objects with commas") {
    JsonWriter json;
    json.begin_object().field("a", 1).key("list").begin_array().value(2).value("x").null().end_array();
    json.key("inner").begin_object().field("flag", true).end_object().end_object();
    CHECK_EQ(json.str(), std::string("{\"a\":1,\"list\":[2,\"x\",null],\"inner\":{\"flag\":true}}"));
}

TEST_CASE("JsonWriter formats doubles compactly and non-finite values as null") {
    JsonWriter json;
    json.begin_array()
        .value(1.5)
        .value(2.0)
        .value(0.12346)
        .value(-0.00001)
        .value(std::numeric_limits<double>::quiet_NaN())
        .value(std::optional<double>())
        .value(std::optional<uint64_t>(7))
        .end_array();
    CHECK_EQ(json.str(), std::string("[1.5,2,0.1235,0,null,null,7]"));
}

TEST_CASE("format_sample aggregates tree totals") {
    Sample sample;
    sample.dt = 1.0;
    ProcessSample first;
    first.pid = 1;
    first.cpu_pct = 50;
    first.rss_kb = 100;
    first.read_bytes_per_s = 10.0;
    ProcessSample second;
    second.pid = 2;
    second.cpu_pct = 25;
    second.rss_kb = 50;
    sample.processes = {first, second};
    std::string line = format_sample(sample, 3, 1.0, 2.0, AgentUsage{});
    CHECK(line.find("\"tree\":{\"nproc\":2,\"cpu_pct\":75,\"rss_kb\":150,\"read_bytes_per_s\":10,\"write_bytes_per_s\":null}") !=
          std::string::npos);
    CHECK(line.find("\"seq\":3") != std::string::npos);
}
