#include "json_writer.h"

#include <cmath>
#include <cstdio>

namespace mlplat {

namespace {

std::size_t utf8_sequence_length(std::string_view text, std::size_t i) {
    auto byte = [&](std::size_t k) { return static_cast<unsigned char>(text[k]); };
    unsigned char lead = byte(i);
    std::size_t length = 0;
    if (lead >= 0xC2 && lead <= 0xDF) length = 2;
    else if (lead >= 0xE0 && lead <= 0xEF) length = 3;
    else if (lead >= 0xF0 && lead <= 0xF4) length = 4;
    else return 0;
    if (i + length > text.size()) return 0;
    for (std::size_t k = 1; k < length; ++k) {
        if ((byte(i + k) & 0xC0) != 0x80) return 0;
    }
    if (lead == 0xE0 && byte(i + 1) < 0xA0) return 0;
    if (lead == 0xED && byte(i + 1) > 0x9F) return 0;
    if (lead == 0xF0 && byte(i + 1) < 0x90) return 0;
    if (lead == 0xF4 && byte(i + 1) > 0x8F) return 0;
    return length;
}

}

std::string escape_json(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 2);
    out.push_back('"');
    for (std::size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c >= 0x80) {
            std::size_t length = utf8_sequence_length(text, i);
            if (length == 0) {
                out += "\\ufffd";
                ++i;
            } else {
                out.append(text.substr(i, length));
                i += length;
            }
            continue;
        }
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char escaped[8];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
                    out += escaped;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
        ++i;
    }
    out.push_back('"');
    return out;
}

void JsonWriter::before_value() {
    if (after_key_) {
        after_key_ = false;
        return;
    }
    if (!needs_comma_.empty()) {
        if (needs_comma_.back()) buffer_.push_back(',');
        needs_comma_.back() = true;
    }
}

JsonWriter& JsonWriter::begin_object() {
    before_value();
    buffer_.push_back('{');
    needs_comma_.push_back(false);
    return *this;
}

JsonWriter& JsonWriter::end_object() {
    buffer_.push_back('}');
    needs_comma_.pop_back();
    return *this;
}

JsonWriter& JsonWriter::begin_array() {
    before_value();
    buffer_.push_back('[');
    needs_comma_.push_back(false);
    return *this;
}

JsonWriter& JsonWriter::end_array() {
    buffer_.push_back(']');
    needs_comma_.pop_back();
    return *this;
}

JsonWriter& JsonWriter::key(std::string_view name) {
    before_value();
    buffer_ += escape_json(name);
    buffer_.push_back(':');
    after_key_ = true;
    return *this;
}

JsonWriter& JsonWriter::value(std::string_view text) {
    before_value();
    buffer_ += escape_json(text);
    return *this;
}

JsonWriter& JsonWriter::value(int64_t number) {
    before_value();
    buffer_ += std::to_string(number);
    return *this;
}

JsonWriter& JsonWriter::value(uint64_t number) {
    before_value();
    buffer_ += std::to_string(number);
    return *this;
}

JsonWriter& JsonWriter::value(double number) {
    if (!std::isfinite(number)) return null();
    before_value();
    char formatted[64];
    std::snprintf(formatted, sizeof(formatted), "%.4f", number);
    std::string_view text(formatted);
    if (text.find('.') != std::string_view::npos) {
        while (text.back() == '0') text.remove_suffix(1);
        if (text.back() == '.') text.remove_suffix(1);
    }
    if (text == "-0") text = "0";
    buffer_.append(text);
    return *this;
}

JsonWriter& JsonWriter::value(bool flag) {
    before_value();
    buffer_ += flag ? "true" : "false";
    return *this;
}

JsonWriter& JsonWriter::null() {
    before_value();
    buffer_ += "null";
    return *this;
}

void JsonWriter::clear() {
    buffer_.clear();
    needs_comma_.clear();
    after_key_ = false;
}

}
