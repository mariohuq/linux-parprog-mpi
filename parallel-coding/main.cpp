//
// Created by mhq on 02/02/23.
//
// start snippet header
#include "huffman.h"
#include "runlength.h"

#include <random>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <mpi.h>

int MASTER_RANK = 0;
const int ALPHABET_SIZE = 25;
// end snippet header

void mpi_encode_huffman(const char* filename);
std::string my_gatherv(std::string suboutput, int rank, int world_size);
std::pair<std::string, int> my_scatter(std::string input, int world_size, int symbol_bytes);
void mpi_encode_rle();
void mpi_decode_rle();
void mpi_generate();

// std::ios_base::sync_with_stdio(false);

// start snippet main
int main(int argc, const char *argv[]) {
    if (argc == 2 && strcmp(argv[1], "generate") == 0) {
        mpi_generate();
        return EXIT_SUCCESS;
    }
    if (argc == 3 && strcmp(argv[1], "encode_huffman") == 0) {
        mpi_encode_huffman(argv[2]);
        return EXIT_SUCCESS;
    }
    if (argc == 2 && strcmp(argv[1], "decode_huffman") == 0) {
        huffman::decode(std::cin, std::cout);
        return EXIT_SUCCESS;
    }
    if (argc == 2 && strcmp(argv[1], "encode_rle") == 0) {
        mpi_encode_rle();
        return EXIT_SUCCESS;
    }
    if (argc == 2 && strcmp(argv[1], "decode_rle") == 0) {
        mpi_decode_rle();
        return EXIT_SUCCESS;
    }

    return EXIT_FAILURE;
}
// end snippet main

void generate_file(const std::string &alphabet,
                   size_t characters_count,
                   std::ostream &os,
                   std::mt19937 &generator) {
    std::uniform_int_distribution<size_t> distribution(0, alphabet.size() - 1);
    for (int i = 0; i < characters_count; ++i) {
        os << alphabet[distribution(generator)];
    }
}

// start snippet mpi_encode_huffman
void mpi_encode_huffman(const char *filename) {
    MPI::Init();
    int rank = MPI::COMM_WORLD.Get_rank();
    int world_size = MPI::COMM_WORLD.Get_size();
    std::string input;
    if (rank == MASTER_RANK) {
        input = (std::ostringstream{} << std::ifstream{filename}.rdbuf()).str();
    };
    std::string subinput;
    subinput.resize(huffman::PART_SIZE);
    MPI::COMM_WORLD.Scatter(input.data(), huffman::PART_SIZE, MPI::CHAR,
                            subinput.data(), huffman::PART_SIZE, MPI::CHAR,
                            MASTER_RANK);
    auto subinput_stream = std::istringstream{subinput};
    auto subfrequencies = huffman::frequencies(subinput_stream);
    std::vector<huffman::freq_t> data(0x100);
    for (auto [k, freq] : subfrequencies) {
        data[static_cast<unsigned char>(k)] = freq;
    }
    MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, data.data(), 0x100, MPI::UNSIGNED_LONG_LONG, MPI::SUM);
    std::map<char, huffman::freq_t> frequencies;
    for (int i = 0; i < 0x100; ++i) {
        if (data[i] == 0) {
            continue;
        }
        frequencies[i] = data[i];
    }
    auto [coding, apriori, longest] = huffman::coding(frequencies);
    size_t body_part_size = 0;
    for (auto [k, freq] : subfrequencies) {
        body_part_size += freq * coding[k].length;
    }

    subinput_stream.clear();
    subinput_stream.seekg(0, std::ios::beg); // rewind

    std::ostringstream suboutput_stream;
    if (rank == MASTER_RANK) {
        huffman::encode_head(std::cout, coding);
    }
    huffman::encode_body(subinput_stream, suboutput_stream, coding, longest, body_part_size);
    auto output = my_gatherv(suboutput_stream.str(), rank, world_size);
    if (rank == MASTER_RANK) {
        std::cout << output;
    }
    if (rank == MASTER_RANK) {
        std::cerr
            << "цена кодирования = "
            << static_cast<double>(apriori.body_size_bits) / static_cast<double>(apriori.message_length) << '\n'
            << "коэффициент сжатия = "
            << static_cast<double>(output.size()) / static_cast<double>(input.size()) << '\n';
    }
    MPI::Finalize();
}
// end snippet mpi_encode_huffman
// start snippet my_gatherv
std::string my_gatherv(std::string suboutput, int rank, int world_size) {
    int size = suboutput.size();
    std::vector<int> sizes;
    if (rank == MASTER_RANK) {
        sizes.resize(world_size);
    }
    MPI::COMM_WORLD.Gather(&size, 1, MPI::INT,
                           sizes.data(), 1, MPI::INT, MASTER_RANK);
    std::vector<int> displacements;
    if (rank == MASTER_RANK) {
        displacements.resize(world_size, 0);
        for (int i = 1; i < world_size; i++) {
            displacements[i] = (displacements[i - 1] + sizes[i - 1]);
        }
    }
    std::string output;
    if (rank == MASTER_RANK) {
        output.resize(displacements.back() + sizes.back());
    }
    MPI::COMM_WORLD.Gatherv(suboutput.data(), suboutput.size(), MPI::CHAR,
                            output.data(), sizes.data(), displacements.data(), MPI::CHAR,
                            MASTER_RANK);
    return output;
}
// end snippet my_gatherv
// start snippet my_scatter
std::pair<std::string, int> my_scatter(std::string input, int world_size, int symbol_bytes) {
    auto [subinput_size, leftover] = std::div(input.size() / symbol_bytes, world_size);
    subinput_size *= symbol_bytes;
    MPI::COMM_WORLD.Bcast(&subinput_size, 1, MPI::INT, MASTER_RANK);
    std::string subinput;
    subinput.resize(subinput_size);
    MPI::COMM_WORLD.Scatter(input.data(),subinput_size, MPI::CHAR,
                            subinput.data(), subinput_size, MPI::CHAR,
                            MASTER_RANK);
    return {subinput, leftover * 2};
}
// end snippet my_scatter
// start snippet mpi_encode_rle
void mpi_encode_rle() {
    MPI::Init();
    int rank = MPI::COMM_WORLD.Get_rank();
    int world_size = MPI::COMM_WORLD.Get_size();

    std::string input;
    if (rank == MASTER_RANK) {
        input = (std::ostringstream{} << std::cin.rdbuf()).str();
    }

    auto [subinput, leftover] = my_scatter(input, world_size, 1);
    std::istringstream subinput_stream { subinput };
    std::ostringstream suboutput_stream;
    auto stats = rle::encode(subinput_stream, suboutput_stream);
    std::string output = my_gatherv(suboutput_stream.str(), rank, world_size);
    if (rank == MASTER_RANK) {
        std::cout << output;
        stats.input_size = input.size();
        stats.output_size = output.size();
    }
    if (rank == MASTER_RANK && leftover != 0) {
        auto is = std::istringstream{ std::string(std::string_view(input).substr(input.size() - leftover, leftover)) };
        auto x = rle::encode(is, std::cout);
        stats.input_size += x.input_size;
        stats.output_size += x.output_size;
    }
    if (rank == MASTER_RANK) {
        std::cerr
            << "коэффициент сжатия = "
            << static_cast<double>(stats.output_size) / static_cast<double>(stats.input_size) << '\n';
    }
    MPI::Finalize();
}
// end snippet mpi_encode_rle
// start snippet mpi_decode_rle
void mpi_decode_rle() {
    MPI::Init();
    int rank = MPI::COMM_WORLD.Get_rank();
    int world_size = MPI::COMM_WORLD.Get_size();
    std::string input;
    if (rank == MASTER_RANK) {
        input = (std::ostringstream{} << std::cin.rdbuf()).str();
    }
    auto [subinput, leftover] = my_scatter(input, world_size, 2);
    std::istringstream subinput_stream { subinput };
    std::ostringstream suboutput_stream;
    rle::decode(subinput_stream, suboutput_stream);
    std::string output = my_gatherv(suboutput_stream.str(), rank, world_size);
    if (rank == MASTER_RANK) {
        std::cout << output;
    }
    if (rank == MASTER_RANK && leftover != 0) {
        auto is = std::istringstream{std::string(std::string_view(input).substr(input.size() - leftover, leftover))};
        rle::decode(is, std::cout);
    }
    MPI::Finalize();
}
// end snippet mpi_decode_rle
// start snippet mpi_generate
void mpi_generate() {
    MPI::Init();
    int rank = MPI::COMM_WORLD.Get_rank();
    int world_size = MPI::COMM_WORLD.Get_size();
    std::mt19937 gen{std::random_device{}()};
    std::string alphabet;
    if (rank == MASTER_RANK) {
        alphabet = (std::ostringstream{} << std::cin.rdbuf()).str();
    } else {
        alphabet.resize(ALPHABET_SIZE);
    }
    MPI::COMM_WORLD.Bcast(alphabet.data(), ALPHABET_SIZE, MPI::CHAR, MASTER_RANK);
    std::stringstream result;
    generate_file(alphabet, huffman::PART_SIZE, result, gen);
    std::string output;
    if (rank == MASTER_RANK) {
        output.resize(world_size * huffman::PART_SIZE);
    }
    MPI::COMM_WORLD.Gather(result.str().data(), huffman::PART_SIZE, MPI::CHAR,
                           output.data(), huffman::PART_SIZE, MPI::CHAR,
                           MASTER_RANK);
    if (rank == MASTER_RANK) {
        std::cout << output;
    }
    MPI::Finalize();
}
// end snippet mpi_generate
