/**
 * @file profiler.hpp
 * @brief Defines performance profiling utilities for the TurboInfer library.
 * @author J.J.G. Pleunes
 */

#pragma once

#include <string>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <cstddef>

namespace turboinfer {
namespace util {

/**
 * @struct ProfileStats
 * @brief Statistics for a profiled operation.
 */
struct ProfileStats {
    std::string name;           ///< Operation name
    size_t call_count = 0;      ///< Number of times called
    double total_time_ms = 0.0; ///< Total execution time in milliseconds
    double min_time_ms = 0.0;   ///< Minimum execution time
    double max_time_ms = 0.0;   ///< Maximum execution time
    double avg_time_ms = 0.0;   ///< Average execution time
    size_t total_memory_bytes = 0; ///< Total memory allocated
};

/**
 * @class Timer
 * @brief High-resolution timer for measuring execution time.
 */
class Timer {
public:
    /**
     * @brief Constructs and starts the timer.
     */
    Timer();

    /**
     * @brief Starts or restarts the timer.
     */
    void start();

    /**
     * @brief Stops the timer and returns elapsed time.
     * @return Elapsed time in milliseconds.
     */
    double stop();

    /**
     * @brief Gets elapsed time without stopping the timer.
     * @return Elapsed time in milliseconds since start.
     */
    double elapsed() const;

    /**
     * @brief Checks if the timer is currently running.
     * @return True if timer is running, false otherwise.
     */
    bool is_running() const noexcept { return running_; }

private:
    std::chrono::high_resolution_clock::time_point start_time_; ///< Timer start time
    bool running_ = false;                                      ///< Whether timer is running
};

/**
 * @class Profiler
 * @brief System for collecting and analyzing performance metrics.
 * 
 * The Profiler class provides tools for measuring execution time, memory usage,
 * and other performance metrics across different operations in TurboInfer.
 */
class Profiler {
public:
    /**
     * @brief Gets the global profiler instance.
     * @return Reference to the global profiler.
     */
    static Profiler& instance();

    /**
     * @brief Enables or disables profiling globally.
     * @param enabled Whether profiling should be active.
     */
    void set_enabled(bool enabled) noexcept { enabled_ = enabled; }

    /**
     * @brief Checks if profiling is enabled.
     * @return True if profiling is enabled, false otherwise.
     */
    bool is_enabled() const noexcept { return enabled_; }

    /**
     * @brief Starts profiling an operation.
     * @param name Operation name.
     * @return Timer ID for stopping the operation.
     */
    size_t start_operation(const std::string& name);

    /**
     * @brief Stops profiling an operation.
     * @param timer_id Timer ID returned from start_operation.
     * @param memory_used Optional memory usage for this operation.
     */
    void stop_operation(size_t timer_id, size_t memory_used = 0);

    /**
     * @brief Records a complete operation timing.
     * @param name Operation name.
     * @param time_ms Execution time in milliseconds.
     * @param memory_used Optional memory usage for this operation.
     */
    void record_operation(const std::string& name, double time_ms, size_t memory_used = 0);

    /**
     * @brief Gets statistics for a specific operation.
     * @param name Operation name.
     * @return Profile statistics, or nullptr if operation not found.
     */
    const ProfileStats* get_stats(const std::string& name) const;

    /**
     * @brief Gets statistics for all operations.
     * @return Map of operation names to statistics.
     */
    std::unordered_map<std::string, ProfileStats> get_all_stats() const;

    /**
     * @brief Clears all profiling data.
     */
    void clear();

    /**
     * @brief Generates a formatted report of all profiling data.
     * @param sort_by_time Whether to sort operations by total time (default: true).
     * @return Formatted profiling report.
     */
    std::string generate_report(bool sort_by_time = true) const;

    /**
     * @brief Saves profiling data to a file.
     * @param file_path Output file path.
     * @param format Output format ("text", "json", "csv").
     * @return True if saved successfully, false otherwise.
     */
    bool save_report(const std::string& file_path, const std::string& format = "text") const;

private:
    bool enabled_ = false;                              ///< Whether profiling is enabled
    std::unique_ptr<class ProfilerImpl> impl_;          ///< Implementation details

    /**
     * @brief Private constructor for singleton pattern.
     */
    Profiler();

    /**
     * @brief Private destructor.
     */
    ~Profiler();

    // Non-copyable and non-movable
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    Profiler(Profiler&&) = delete;
    Profiler& operator=(Profiler&&) = delete;
};

/**
 * @class ScopedProfiler
 * @brief RAII-style profiler for automatically timing scoped operations.
 */
class ScopedProfiler {
public:
    /**
     * @brief Constructs a scoped profiler and starts timing.
     * @param name Operation name.
     */
    explicit ScopedProfiler(const std::string& name);

    /**
     * @brief Destructor that automatically stops timing.
     */
    ~ScopedProfiler();

    /**
     * @brief Sets the memory usage for this operation.
     * @param memory_bytes Memory usage in bytes.
     */
    void set_memory_usage(size_t memory_bytes) { memory_used_ = memory_bytes; }

private:
    size_t timer_id_;           ///< Timer ID from profiler
    size_t memory_used_ = 0;    ///< Memory usage for this operation

    // Non-copyable and non-movable
    ScopedProfiler(const ScopedProfiler&) = delete;
    ScopedProfiler& operator=(const ScopedProfiler&) = delete;
    ScopedProfiler(ScopedProfiler&&) = delete;
    ScopedProfiler& operator=(ScopedProfiler&&) = delete;
};

/**
 * @class MemoryProfiler
 * @brief Utility for tracking memory usage patterns.
 */
class MemoryProfiler {
public:
    /**
     * @brief Records memory allocation.
     * @param size Size of allocation in bytes.
     * @param tag Optional tag for categorizing allocation.
     */
    static void record_allocation(size_t size, const std::string& tag = "");

    /**
     * @brief Records memory deallocation.
     * @param size Size of deallocation in bytes.
     * @param tag Optional tag for categorizing deallocation.
     */
    static void record_deallocation(size_t size, const std::string& tag = "");

    /**
     * @brief Gets current memory usage by tag.
     * @param tag Memory tag to query.
     * @return Current memory usage in bytes.
     */
    static size_t get_memory_usage(const std::string& tag = "");

    /**
     * @brief Gets peak memory usage by tag.
     * @param tag Memory tag to query.
     * @return Peak memory usage in bytes.
     */
    static size_t get_peak_memory_usage(const std::string& tag = "");

    /**
     * @brief Resets all memory tracking data.
     */
    static void reset();

    /**
     * @brief Generates a memory usage report.
     * @return Formatted memory usage report.
     */
    static std::string generate_report();
};

} // namespace util
} // namespace turboinfer

// Convenience macros for profiling
#define TURBOINFER_PROFILE(name) \
    ::turboinfer::util::ScopedProfiler _profiler_##__LINE__(name)

#define TURBOINFER_PROFILE_FUNCTION() \
    TURBOINFER_PROFILE(__FUNCTION__)

#define TURBOINFER_PROFILE_SCOPE(name) \
    TURBOINFER_PROFILE(name)

// Memory profiling macros
#define TURBOINFER_RECORD_ALLOC(size, tag) \
    ::turboinfer::util::MemoryProfiler::record_allocation(size, tag)

#define TURBOINFER_RECORD_DEALLOC(size, tag) \
    ::turboinfer::util::MemoryProfiler::record_deallocation(size, tag)
