//
// Created by mhq on 05/02/23.
//
#pragma once
#include <cstddef>
#include <tuple>
#include <istream>
#include <ostream>
#include <map>

constexpr size_t L = 24; // max code length for alphabet of 25 chars
struct Code {
    unsigned int value: L;
    unsigned char length;

    Code() = default;

    bool operator==(const Code other) const {
        return std::tie(value, length) == std::tie(other.value, other.length);
    }
    bool operator<(const Code other) const {
        return std::tie(value, length) < std::tie(other.value, other.length);
    }

    [[nodiscard]] Code with_zero() const {
        return {value,static_cast<decltype(length)>(length + 1u)};
    }

    [[nodiscard]] Code with_one() const {
        return {value | (1u << length),static_cast<decltype(length)>(length + 1u)};
    }
};
static_assert(sizeof(Code) == 4);

struct EncodingStats {
    size_t output_size; // bytes
    size_t input_size; // bytes
};

struct AprioriStats {
    size_t body_size_bits;
    size_t message_length;
};

namespace huffman {
    using freq_t = size_t;
    using FrequencyMap = std::map<char, freq_t>;
    using AlphabetCoding = std::map<char, Code>;
    using AlphabetDecoding = std::map<Code, char>;

    const int PART_SIZE = 10'000;

    struct Coding {
        AlphabetCoding coding;
        AprioriStats apriori;
        Code longest;
    };

    FrequencyMap frequencies(std::istream& is);
    Coding coding(FrequencyMap freqs);
    void encode_head(std::ostream &os, const AlphabetCoding &coding);
    EncodingStats encode_body(std::istream &is, std::ostream &os, const AlphabetCoding &coding, Code longest, size_t body_size_bits);

    EncodingStats my_encode(std::istream &is, std::ostream &os, const AlphabetCoding &coding, Code longest, size_t body_size_bits);
    std::pair<AprioriStats, EncodingStats> encode(std::istream& is, std::ostream& os);
    AlphabetDecoding decode_head(std::istream& is);
    void decode_body(const AlphabetDecoding& decoding, std::istream& is, std::ostream& os);
    void decode(std::istream& is, std::ostream& os);
}

void test_huffman();
void test_header();
