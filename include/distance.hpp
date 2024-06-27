#pragma once
#include <span>

namespace pickle
{
    template<class T, std::size_t Extend = std::dynamic_extent>
    float L1(const std::span<T, Extend> va, const std::span<T, Extend> vb);
    
    template<class T, std::size_t Extend = std::dynamic_extent>
    float L2(const std::span<T, Extend> va, const std::span<T, Extend> vb);

    template<class T, std::size_t Extend = std::dynamic_extent>
    float Ip(const std::span<T, Extend> va, const std::span<T, Extend> vb);
}