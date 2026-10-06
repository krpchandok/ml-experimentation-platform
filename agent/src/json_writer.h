#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mlplat {

class JsonWriter {
public:
    JsonWriter& begin_object();
    JsonWriter& end_object();
    JsonWriter& begin_array();
    JsonWriter& end_array();
    JsonWriter& key(std::string_view name);
    JsonWriter& value(std::string_view text);
    JsonWriter& value(const char* text) { return value(std::string_view(text)); }
    JsonWriter& value(const std::string& text) { return value(std::string_view(text)); }
    JsonWriter& value(int64_t number);
    JsonWriter& value(uint64_t number);
    JsonWriter& value(int number) { return value(static_cast<int64_t>(number)); }
    JsonWriter& value(double number);
    JsonWriter& value(bool flag);
    JsonWriter& null();

    template <typename T>
    JsonWriter& value(const std::optional<T>& maybe) {
        return maybe ? value(*maybe) : null();
    }

    template <typename T>
    JsonWriter& field(std::string_view name, const T& item) {
        key(name);
        return value(item);
    }

    const std::string& str() const { return buffer_; }
    void clear();

private:
    void before_value();

    std::string buffer_;
    std::vector<bool> needs_comma_;
    bool after_key_ = false;
};

std::string escape_json(std::string_view text);

}
