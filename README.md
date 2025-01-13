# Pickle

HNSW revisited.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Benchmark](#Benchmark)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About

Pickle is an implementation of HNSW index that supports parallel graph indexing and search. 

## Features

Key features of pickle:
- Efficient multi-threaded graph construction.
- Achieve similar performance compared to other open-sourced solutions like [HNSWlib](https://github.com/nmslib/hnswlib) and [FAISS](https://github.com/facebookresearch/faiss).
- Support multiple profiling strategies (query per second, number of distance computation, etc).
- Header only library.

## Benchmark:

We evalute the performance of pickle on a server class machine equipped with dual Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz (12 cores / 24 threads each). The server has 384GB of RAM.

We compare pickle with FAISS and HNSWlib in terms of query per second (QPS) and the number of distance caculation required to reach the same level of recall. You can find the results in [plot/figs](plot/figs) and the detailed log in [plot/log.csv](plot/log.csv).

## Getting Started

### Prerequisites

List the software and libraries needed to run your project:
- C++ compiler (e.g., GCC, Clang, MSVC)
- CMake
- [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&linux-install=apt) (FAISS requires this library)

Install MKL on Ubuntu (needs 6GB disk space):
```bash
cd /tmp
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/79153e0f-74d7-45af-b8c2-258941adf58a/intel-onemkl-2025.0.0.940_offline.sh
sh intel-onemkl-2025.0.0.940_offline.sh -a -s --eula accept
# The default install directory is ~/intel
```

### Installation

Step-by-step guide to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/juelinl/pickle.git
   cd pickle
   git submodule update --init --recursive
   ```
2. Build the project (if using CMake):
   ```bash
   mkdir build
   pushd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build ./ -j
   popd
   ```
4. Run the program:
   ```bash
   ./build/benchmark/hnsw_pickle -h
   ```

5. For a more detailed usage, checkout the example script `bench.sh`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was heavily inspired by the incredible work behind [HNSWlib](https://github.com/nmslib/hnswlib) and [FAISS](https://github.com/facebookresearch/faiss). Both libraries have set high standards for efficient and scalable similarity search, and their implementations served as a foundation for many of the ideas explored in this project.

We would like to extend our gratitude to the creators and maintainers of these libraries for their contributions to the open-source community. Their dedication to building high-performance tools has significantly advanced the field of approximate nearest neighbor search and vector similarity search.