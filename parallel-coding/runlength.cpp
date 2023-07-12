//
// Created by mhq on 27/02/23.
//
#include "runlength.h"
#include <cassert>
#include <cstdint>

struct Code {
    char byte;
    uint8_t count;
};

rle::EncodingStats rle::encode(std::istream& is, std::ostream& os) {
    Code current;
    rle::EncodingStats stats{};
    size_t overbytes;
    while (is.get(current.byte)) {
        current.count = 1;
        overbytes = 0;
        while (is.peek() == current.byte) {
            current.count++;
            if (current.count == 0) {
                overbytes++;
            }
            is.get();
        }
        if (overbytes > 0) {
            Code over = {current.byte, 0};
            for (size_t i = 0; i < overbytes; ++i) {
                os.write(reinterpret_cast<char*>(&over), sizeof(over));
                stats.output_size += 2;
                stats.input_size += 0x100;
            }
            if (current.count == 0) {
                continue;
            }
        }
        os.write(reinterpret_cast<char*>(&current), sizeof(current));
        stats.output_size += 2;
        stats.input_size += current.count;
    }
    return stats;
}

void rle::decode(std::istream& is, std::ostream& os) {
    Code current;
    while(is.read(reinterpret_cast<char*>(&current), sizeof(current))) {
        assert(is.gcount() == sizeof(current));
        for (int i = 0; i < current.count; ++i) {
            os.put(current.byte);
        }
    }
}