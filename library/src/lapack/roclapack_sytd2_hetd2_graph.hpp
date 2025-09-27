#pragma once

#include "roclapack_sytd2_hetd2.hpp"
#include <hip/hip_runtime.h>
#include <unordered_map>

ROCSOLVER_BEGIN_NAMESPACE

// HIP Graph-based optimization for SYTD2/HETD2
// This captures kernel sequences and replays them to reduce launch overhead
template <typename T, typename S>
class SYTD2GraphOptimizer
{
private:
    struct GraphKey
    {
        rocblas_int n;
        rocblas_fill uplo;
        rocblas_int batch_count;

        bool operator==(const GraphKey& other) const
        {
            return n == other.n && uplo == other.uplo && batch_count == other.batch_count;
        }
    };

    struct GraphKeyHash
    {
        std::size_t operator()(const GraphKey& key) const
        {
            return std::hash<int>()(key.n) ^ (std::hash<int>()(key.uplo) << 1) ^
                   (std::hash<int>()(key.batch_count) << 2);
        }
    };

    struct GraphInstance
    {
        hipGraph_t graph;
        hipGraphExec_t exec;
        bool captured;

        GraphInstance() : graph(nullptr), exec(nullptr), captured(false) {}

        ~GraphInstance()
        {
            if(captured)
            {
                hipGraphExecDestroy(exec);
                hipGraphDestroy(graph);
            }
        }
    };

    std::unordered_map<GraphKey, std::unique_ptr<GraphInstance>, GraphKeyHash> graph_cache;
    static constexpr int MAX_CACHE_SIZE = 32; // Limit cache size

public:
    rocblas_status execute_with_graph(rocblas_handle handle,
                                      const rocblas_fill uplo,
                                      const rocblas_int n,
                                      T* A,
                                      const rocblas_int shiftA,
                                      const rocblas_int lda,
                                      const rocblas_stride strideA,
                                      S* D,
                                      const rocblas_stride strideD,
                                      S* E,
                                      const rocblas_stride strideE,
                                      T* tau,
                                      const rocblas_stride strideP,
                                      const rocblas_int batch_count,
                                      T* scalars,
                                      T* work,
                                      T* norms,
                                      T* tmptau,
                                      T** workArr)
    {
        // Only use graphs for common sizes to avoid memory explosion
        if(n < 256 || n > 2048 || batch_count > 10)
        {
            // Fall back to regular implementation
            return rocsolver_sytd2_hetd2_template<T>(handle, uplo, n, A, shiftA, lda, strideA,
                                                     D, strideD, E, strideE, tau, strideP,
                                                     batch_count, scalars, work, norms, tmptau, workArr);
        }

        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        GraphKey key{n, uplo, batch_count};

        // Check cache
        auto it = graph_cache.find(key);
        if(it == graph_cache.end())
        {
            // Clean cache if too large
            if(graph_cache.size() >= MAX_CACHE_SIZE)
            {
                graph_cache.clear();
            }

            // Create new graph instance
            auto instance = std::make_unique<GraphInstance>();

            // Begin capture
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);

            // Execute the function to capture
            rocblas_status status = rocsolver_sytd2_hetd2_template<T>(
                handle, uplo, n, A, shiftA, lda, strideA,
                D, strideD, E, strideE, tau, strideP,
                batch_count, scalars, work, norms, tmptau, workArr);

            // End capture
            hipGraph_t graph;
            hipStreamEndCapture(stream, &graph);

            // Create executable graph
            hipGraphExec_t exec;
            hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

            instance->graph = graph;
            instance->exec = exec;
            instance->captured = true;

            // Store in cache
            auto result = graph_cache.emplace(key, std::move(instance));
            it = result.first;
        }

        // Launch the cached graph
        hipGraphLaunch(it->second->exec, stream);

        return rocblas_status_success;
    }

    // Clear cache when needed
    void clear_cache()
    {
        graph_cache.clear();
    }
};

// Global instance for caching graphs
static thread_local SYTD2GraphOptimizer<float, float> sytd2_graph_optimizer_float;
static thread_local SYTD2GraphOptimizer<double, double> sytd2_graph_optimizer_double;
static thread_local SYTD2GraphOptimizer<rocblas_float_complex, float> hetd2_graph_optimizer_float;
static thread_local SYTD2GraphOptimizer<rocblas_double_complex, double> hetd2_graph_optimizer_double;

// Wrapper function to use graph optimization
template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_graph_optimized(rocblas_handle handle,
                                                     const rocblas_fill uplo,
                                                     const rocblas_int n,
                                                     U A,
                                                     const rocblas_int shiftA,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     S* D,
                                                     const rocblas_stride strideD,
                                                     S* E,
                                                     const rocblas_stride strideE,
                                                     T* tau,
                                                     const rocblas_stride strideP,
                                                     const rocblas_int batch_count,
                                                     T* scalars,
                                                     T* work,
                                                     T* norms,
                                                     T* tmptau,
                                                     T** workArr)
{
    // Select the appropriate optimizer based on type
    if constexpr(std::is_same_v<T, float>)
    {
        return sytd2_graph_optimizer_float.execute_with_graph(
            handle, uplo, n, (float*)A, shiftA, lda, strideA,
            D, strideD, E, strideE, tau, strideP, batch_count,
            scalars, work, norms, tmptau, workArr);
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return sytd2_graph_optimizer_double.execute_with_graph(
            handle, uplo, n, (double*)A, shiftA, lda, strideA,
            D, strideD, E, strideE, tau, strideP, batch_count,
            scalars, work, norms, tmptau, workArr);
    }
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
    {
        return hetd2_graph_optimizer_float.execute_with_graph(
            handle, uplo, n, (rocblas_float_complex*)A, shiftA, lda, strideA,
            D, strideD, E, strideE, tau, strideP, batch_count,
            scalars, work, norms, tmptau, workArr);
    }
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
    {
        return hetd2_graph_optimizer_double.execute_with_graph(
            handle, uplo, n, (rocblas_double_complex*)A, shiftA, lda, strideA,
            D, strideD, E, strideE, tau, strideP, batch_count,
            scalars, work, norms, tmptau, workArr);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE