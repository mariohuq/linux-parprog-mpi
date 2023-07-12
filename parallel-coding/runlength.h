#pragma once
#include <ostream>
#include <istream>

namespace rle {
    struct EncodingStats {
        size_t output_size; // bytes
        size_t input_size; // bytes
    };
    EncodingStats encode(std::istream& is, std::ostream& os);
    void decode(std::istream& is, std::ostream& os);
}