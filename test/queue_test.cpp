//
// Created by juelin on 7/10/24.
//
#include "queue.hpp"
#include <iostream>
using namespace pickle;

std::string parse_entry(Entry e) {
    std::string ret = "vid=";
    ret += std::to_string(e._vid);
    ret += " distance=" + std::to_string(int(e._distance));
    return ret;
}

int main(int argc, const char* argv[]){
    size_t N = 10;
    auto min_queue = EntryHeap<MinFirst>(N);
    MinFirst comp = MinFirst();
    std::cout << "sizeof min comp: " << sizeof(comp) << '\n';
    std::vector<Entry> entries;
    for (size_t i = 0; i < N; i++){
        entries.emplace_back(static_cast<distance_t>(rand() % 1000), rand() % 1000);
    }

    for (auto e : entries) {
        min_queue.insert(e);
    }
    auto res = min_queue.extractTopK(N);
    for (auto e: res) {
        std::cout << parse_entry(e) << "\n";
    }
}