//
// Created by mhq on 05/02/23.
//

#include "huffman.h"

#include <sstream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

using namespace huffman;

AprioriStats coding_price(const std::vector<Code>& codes, const std::vector<freq_t>& frequencies);
EncodingStats encode(std::istream& is, std::ostream& os, const AlphabetCoding& coding, Code longest, size_t body_size_bits);

struct Huffman {
    std::vector<freq_t> probabilities;
    std::vector<Code> codes;
    explicit Huffman(const std::vector<freq_t>& probabilities) : probabilities{probabilities} {
        codes.reserve(probabilities.size());
        Huff();
    }
    std::vector<Code> operator()() const {
        return codes;
    }
private:
     void Huff() {
        size_t n = probabilities.size();
        if (n == 2) {
            codes.push_back(Code{}.with_zero());
            codes.push_back(Code{}.with_one());
            return;
        }
        long j = Up(probabilities[n - 2] + probabilities[n - 1]);
        Huff();
        Down(j);
    }
    long Up(freq_t q) {
        probabilities.pop_back();
        probabilities.pop_back();
        auto where = std::lower_bound(probabilities.rbegin(), probabilities.rend(), q).base();
        return probabilities.insert(where, q) - probabilities.begin();
    }
    void Down(long j) {
        Code current = codes[j];
        codes.erase(codes.begin() + j);
        codes.push_back(current.with_zero());
        codes.push_back(current.with_one());
    }
};

std::pair<AprioriStats, EncodingStats> huffman::encode(std::istream& is, std::ostream& os) {
    FrequencyMap freqs = huffman::frequencies(is);
    auto [coding, apriori, longest] = huffman::coding(freqs);
    is.clear();
    is.seekg(0, std::ios::beg); // rewind
    return {apriori, my_encode(is, os, coding, longest, apriori.body_size_bits)};
}

AlphabetDecoding huffman::decode_head(std::istream& is) {
    size_t alphabet_size;
    if (!is.read(reinterpret_cast<char*>(&alphabet_size), sizeof(size_t))) {
        return {};
    }
    AlphabetDecoding decoding;
    for (; alphabet_size > 0;--alphabet_size) {
        char ch;
        Code c{};
        if (!is.get(ch)) {
            return {};
        }
        if (!is.read(reinterpret_cast<char*>(&c), sizeof(Code))) {
            return {};
        }
        decoding.emplace(c, ch);
    }
    return decoding;
}

void huffman::decode_body(const AlphabetDecoding& decoding, std::istream& is, std::ostream& os) {
    Code currentCode;
    char input_buffer;
    size_t output_index = 0;
    int i;
    int padding_size;
    if (!is.get(input_buffer)) {
        return;
    }
    do {
        if (output_index == 0) {
            Code padding{};
            for (i = 0; i < 2; ++i, input_buffer <<= 1) {
                if (input_buffer < 0) {
                    padding = padding.with_one();
                } else {
                    padding = padding.with_zero();
                }
            }
            padding_size = padding.value == 0 ? 0 : 4 + padding.value;
            currentCode = {};
        }
        int i_max = is.peek() != EOF ? 8 : 8 - padding_size;
        for (; i < i_max; ++i, input_buffer <<= 1) {
            if (input_buffer < 0) {
                currentCode = currentCode.with_one();
            } else {
                currentCode = currentCode.with_zero();
            }
            auto it = decoding.find(currentCode);
            if (it == decoding.end()) {
                continue;
            }
            os << it->second;
            currentCode = {};
            output_index++;
            output_index %= PART_SIZE;
            if (output_index == 0) {
                break;
            }
        }
        i = 0;
    } while(is.get(input_buffer));
}

void huffman::decode(std::istream& is, std::ostream& os) {
    auto decoding = decode_head(is);
    if (decoding.empty()) {
        return;
    }
    decode_body(decoding, is, os);
}

FrequencyMap huffman::frequencies(std::istream &is) {
    FrequencyMap result{};
    for (char c; is.get(c); ) {
        ++result[c];
    }
    return result;
}

Coding huffman::coding(FrequencyMap freqs) {
    std::vector<char> alphabet;
    std::vector<freq_t> frequencies;
    {
        alphabet.reserve(freqs.size());
        for (auto [c, _]: freqs) {
            alphabet.push_back(c);
        }
        std::sort(alphabet.begin(), alphabet.end(), [&](auto left, auto right) {
            return freqs.at(left) > freqs.at(right);
        });

        frequencies.reserve(alphabet.size());
        std::transform(alphabet.begin(), alphabet.end(), std::back_inserter(frequencies),
                       [&](char alpha) {
                           return freqs[alpha];
                       });
    }
    auto codes = Huffman{frequencies}();
    AlphabetCoding encoding{};
    std::transform(alphabet.begin(), alphabet.end(), codes.begin(),
                   std::inserter(encoding, encoding.end()),
                   std::make_pair<const char&, const Code&>);
    return {encoding, coding_price(codes, frequencies), codes.back()};
}

AprioriStats coding_price(const std::vector<Code> &codes, const std::vector<freq_t> &frequencies) {
    return {
            std::inner_product(codes.begin(), codes.end(), frequencies.begin(),
                               size_t{}, std::plus<size_t>{}, [](Code code, freq_t freq) {
                        return code.length * freq;
                    }),
            std::accumulate(frequencies.begin(), frequencies.end(), size_t{})
    };
}

void huffman::encode_head(std::ostream &os, const AlphabetCoding &coding) {
    // Header
    size_t alphabet_size = coding.size();
    os.write(reinterpret_cast<char*>(&alphabet_size), sizeof(size_t));
    for (auto [character, code]: coding) {
        os.put(character);
        os.write(reinterpret_cast<char*>(&code), sizeof(Code));
    }
}

EncodingStats huffman::encode_body(std::istream &is, std::ostream &os, const AlphabetCoding &coding, Code longest, size_t body_size_bits) {
    size_t alphabet_size = coding.size();
    EncodingStats result{};
    result.output_size = sizeof(alphabet_size) + alphabet_size * (sizeof(char) + sizeof(Code));
    size_t nbits = 0;
    char current_byte = '\0';
    auto bitout = [&](Code code) {
        for (; code.length > 0; code.length--, code.value >>= 1) {
            current_byte <<= 1;
            current_byte |= code.value & 1;
            nbits++;
            if (nbits == 8) {
                os.put(current_byte);
                result.output_size++;
                nbits = 0;
                current_byte = '\0';
            }
        }
    };
    // Size of padding at the end
    unsigned int leftover = (body_size_bits + 2) % 8;
    bitout({(leftover == 0 || longest.length > 8 - leftover) ? 0u : (8 - leftover) & 0x11, 2});
    // Body
    char input_buffer;
    result.input_size = 0;
    while (is.get(input_buffer)) {
        bitout(coding.at(input_buffer));
        result.input_size++;
    }
    //Padding
    // use the trick introduced in https://cs.stackexchange.com/a/100163 by @evgeniy-berezovsky
    // and use two bits at the start to know padding size if it in range 5..7 bits
    if (nbits == 0) {
        return result;
    }
    assert(leftover == nbits);
    auto needed_length = 8 - nbits;
    bitout({longest.length > needed_length ? longest.value : 0u, static_cast<unsigned char>(8 - nbits)});
    assert(nbits == 0);
    return result;
}

EncodingStats huffman::my_encode(std::istream &is, std::ostream &os, const AlphabetCoding &coding, Code longest, size_t body_size_bits) {
    encode_head(os, coding);
    return encode_body(is, os, coding, longest, body_size_bits);
}

void test_huffman() {
    std::vector<Code> expected{
            {0b00,     2},
            {0b01,     2},
            {0b11,     2},
            {0b110,    3},
            {0b0010,   4},
            {0b01010,  5},
            {0b011010, 6},
            {0b111010, 6}};
    std::vector<Code> actual = Huffman{{31, 24, 17, 11, 9, 5, 2, 1}}();
    assert(actual == expected);
}

void test_header() {
    AlphabetCoding table{
            {'a', {0b00,     2}},
            {'b', {0b01,     2}},
            {'c', {0b11,     2}},
            {'d', {0b110,    3}},
            {'e', {0b0010,   4}},
            {'f', {0b01010,  5}},
            {'g', {0b011010, 6}},
            {'h', {0b111010, 6}}
    };
    constexpr auto subject= "abcdefgh";
    std::istringstream raw{subject};
    std::stringstream coded;
    my_encode(raw, coded, table, table['h'], 30);
    coded.seekg(0, std::ios::beg);
    std::stringstream result;
    huffman::decode(coded, result);
    std::string output = result.str();
    assert(output == subject);
}