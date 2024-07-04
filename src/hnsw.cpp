//#include "hnsw.hpp"
//
//#include <cstdint>
//
//namespace pickle {
//
//#define BUILD_TEMPLATE_EXPAND(F, DType) \
//    template<> DynamicNSWGraphPtr F<DType, std::dynamic_extent>(DistanceFunction df, size_t ef, size_t max_degree, size_t dim, const std::vector<external_id_t> &external_ids, std::span<DType> all_data);
//
//    BUILD_TEMPLATE_EXPAND(BuildNSWLayer, float) std::span<T>;
//
//    BUILD_TEMPLATE_EXPAND(BuildNSWLayer, uint8_t) std::span<T>;
//
//    BUILD_TEMPLATE_EXPAND(BuildNSWLayer, float16_t) std::span<T>;
//
//    BUILD_TEMPLATE_EXPAND(BuildNSWLayer, int8_t) std::span<T>;
//};