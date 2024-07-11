#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include "common.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include "dataloader.hpp"

namespace pickle {

    std::string getFileExtension(const std::string &filename) {
        size_t dotPos = filename.find_last_of('.');
        if (dotPos == std::string::npos) {
            throw std::runtime_error("File has no extension.");
        }
        return filename.substr(dotPos + 1);
    };

    bool ends_with(std::string const &value, std::string const &ending) {
        if (ending.size() > value.size())
            return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    }

    GroundTruthPtr LoadGroundTruth(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return {};
        }

        // Open the binary file
        GroundTruthPtr ret = std::make_shared<GroundTruth>();
        // Read number of rows and columns
        uint32_t rows, cols;
        // Read rows and cols
        file.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));
        file.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));
        ret->_shape[0] = rows;
        ret->_shape[1] = cols;
        size_t num_elements = ret->_shape[0] * ret->_shape[1];
        ret->_label.resize(num_elements);
        ret->_distance.resize(num_elements);
        file.read(reinterpret_cast<char *>(ret->_label.data()),
                  num_elements * sizeof(uint32_t));
        file.read(reinterpret_cast<char *>(ret->_distance.data()),
                  num_elements * sizeof(float));
        file.close();
        return ret;
    };

    Array2DPtr LoadArray2D(const std::string &filename, size_t max_elements,
                           bool to_float) {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            exit(-1);
        }

        // Open the binary file
        Array2DPtr ret = std::make_shared<Array2D>();
        // Read number of rows and columns
        uint32_t rows, cols;
        // Read rows and cols
        file.read(reinterpret_cast<char *>(&rows), sizeof(int));
        file.read(reinterpret_cast<char *>(&cols), sizeof(int));
        rows = std::min((size_t) rows, max_elements);
        ret->_shape[0] = rows;
        ret->_shape[1] = cols;
        // Determine the type of the matrix elements from the file extension
        std::string extension = getFileExtension(filename);
        if (to_float) {
            ret->_element_size = 4;
            ret->_data_type = DataType::Float32;
            if (extension == "fbin") {
                size_t num_bytes = ret->_element_size * ret->_shape[0] * ret->_shape[1];
                ret->_data = new unsigned char[num_bytes];
                file.read(ret->data<char>(), num_bytes);
                file.close();
                return ret;
            } else if (extension == "u8bin") {
                size_t arr_size = ret->_shape[0] * ret->_shape[1];
                std::vector<uint8_t> data(arr_size);
                file.read((char *) data.data(), arr_size * sizeof(uint8_t));
                file.close();

                size_t num_bytes = ret->_element_size * ret->_shape[0] * ret->_shape[1];
                ret->_data = new unsigned char[num_bytes];
                auto *ptr = reinterpret_cast<float *>(ret->_data);
                for (size_t i = 0; i < data.size(); i++) {
                    ptr[i] = static_cast<float>(data.at(i));
                }
                return ret;
            } else if (extension == "i8bin") {
                size_t arr_size = ret->_shape[0] * ret->_shape[1];
                std::vector<int8_t> data(arr_size);
                file.read((char *) data.data(), arr_size * sizeof(int8_t));
                file.close();

                size_t num_bytes = ret->_element_size * ret->_shape[0] * ret->_shape[1];
                ret->_data = new unsigned char[num_bytes];
                auto *ptr = reinterpret_cast<float *>(ret->_data);
                for (size_t i = 0; i < data.size(); i++) {
                    ptr[i] = static_cast<float>(data.at(i));
                }
                return ret;
            } else if (extension == "f16bin") {
                size_t arr_size = ret->_shape[0] * ret->_shape[1];
                std::vector<float16_t> data(arr_size);
                file.read((char *) data.data(), arr_size * sizeof(_Float16));
                file.close();

                size_t num_bytes = ret->_element_size * ret->_shape[0] * ret->_shape[1];
                ret->_data = new unsigned char[num_bytes];
                auto *ptr = reinterpret_cast<float *>(ret->_data);
                for (size_t i = 0; i < data.size(); i++) {
                    ptr[i] = static_cast<float>(data.at(i));
                }
                return ret;
            } else {
                std::cerr << "Unsupported extension type " << extension << std::endl;
                exit(-1);
            }
        } else {
            if (extension == "fbin") {
                ret->_element_size = 4;
                ret->_data_type = DataType::Float32;
            } else if (extension == "u8bin") {
                ret->_element_size = 1;
                ret->_data_type = DataType::Uint8;
            } else if (extension == "i8bin") {
                ret->_element_size = 1;
                ret->_data_type = DataType::Int8;
            } else if (extension == "f16bin") {
                ret->_element_size = 2;
                ret->_data_type = DataType::Float16;
            } else {
                std::cerr << "Unsupported extension type " << extension << std::endl;
                exit(-1);
            }
            size_t num_bytes = ret->_element_size * ret->_shape[0] * ret->_shape[1];
            ret->_data = new unsigned char[num_bytes];
            file.read(ret->data<char>(), num_bytes);
            file.close();
            return ret;
        }
    }
} // namespace pickle