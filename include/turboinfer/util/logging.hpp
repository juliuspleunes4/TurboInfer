/**
 * @file logging.hpp
 * @brief Defines logging utilities for the TurboInfer library.
 * @author J.J.G. Pleunes
 */

#pragma once

#include <string>
#include <memory>
#include <sstream>

namespace turboinfer {
namespace util {

/**
 * @enum LogLevel
 * @brief Supported logging levels.
 */
enum class LogLevel {
    kDebug = 0,     ///< Detailed debugging information
    kInfo = 1,      ///< General information messages
    kWarning = 2,   ///< Warning messages
    kError = 3,     ///< Error messages
    kFatal = 4      ///< Fatal error messages
};

/**
 * @class Logger
 * @brief Thread-safe logging system with configurable output and formatting.
 */
class Logger {
public:
    /**
     * @brief Gets the global logger instance.
     * @return Reference to the global logger.
     */
    static Logger& instance();

    /**
     * @brief Sets the minimum log level.
     * @param level Minimum level for messages to be logged.
     */
    void set_level(LogLevel level) noexcept { level_ = level; }

    /**
     * @brief Gets the current log level.
     * @return Current minimum log level.
     */
    LogLevel level() const noexcept { return level_; }

    /**
     * @brief Enables or disables console output.
     * @param enabled Whether to output to console.
     */
    void set_console_output(bool enabled) noexcept { console_output_ = enabled; }

    /**
     * @brief Sets the output file for logging.
     * @param file_path Path to log file (empty string disables file output).
     * @return True if file was opened successfully, false otherwise.
     */
    bool set_file_output(const std::string& file_path);

    /**
     * @brief Enables or disables timestamps in log messages.
     * @param enabled Whether to include timestamps.
     */
    void set_timestamp_enabled(bool enabled) noexcept { timestamp_enabled_ = enabled; }

    /**
     * @brief Logs a message at the specified level.
     * @param level Log level for the message.
     * @param message Message to log.
     * @param file Source file name (typically __FILE__).
     * @param line Source line number (typically __LINE__).
     */
    void log(LogLevel level, const std::string& message, 
             const char* file = "", int line = 0);

    /**
     * @brief Flushes all pending log output.
     */
    void flush();

private:
    LogLevel level_ = LogLevel::kInfo;      ///< Current log level
    bool console_output_ = true;            ///< Whether to output to console
    bool timestamp_enabled_ = true;         ///< Whether to include timestamps
    std::unique_ptr<class LoggerImpl> impl_; ///< Implementation details

    /**
     * @brief Private constructor for singleton pattern.
     */
    Logger();

    /**
     * @brief Private destructor.
     */
    ~Logger();

    // Non-copyable and non-movable
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;
};

/**
 * @class LogStream
 * @brief Stream-based logging interface for convenient message formatting.
 */
class LogStream {
public:
    /**
     * @brief Constructs a log stream for the specified level.
     * @param level Log level.
     * @param file Source file name.
     * @param line Source line number.
     */
    LogStream(LogLevel level, const char* file, int line);

    /**
     * @brief Destructor that outputs the accumulated message.
     */
    ~LogStream();

    /**
     * @brief Stream insertion operator for various types.
     * @tparam T Type of value to insert.
     * @param value Value to insert into the stream.
     * @return Reference to this stream.
     */
    template<typename T>
    LogStream& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

private:
    LogLevel level_;            ///< Log level for this message
    const char* file_;          ///< Source file name
    int line_;                  ///< Source line number
    std::ostringstream stream_; ///< Message accumulator
};

/**
 * @brief Converts log level to string representation.
 * @param level Log level.
 * @return String representation of the log level.
 */
const char* log_level_to_string(LogLevel level);

/**
 * @brief Converts string to log level.
 * @param str String representation of log level.
 * @return Corresponding log level.
 * @throws std::invalid_argument if string is not a valid log level.
 */
LogLevel string_to_log_level(const std::string& str);

} // namespace util
} // namespace turboinfer

// Convenience macros for logging
#define TURBOINFER_LOG(level) \
    ::turboinfer::util::LogStream(level, __FILE__, __LINE__)

#define TURBOINFER_LOG_DEBUG() \
    TURBOINFER_LOG(::turboinfer::util::LogLevel::kDebug)

#define TURBOINFER_LOG_INFO() \
    TURBOINFER_LOG(::turboinfer::util::LogLevel::kInfo)

#define TURBOINFER_LOG_WARNING() \
    TURBOINFER_LOG(::turboinfer::util::LogLevel::kWarning)

#define TURBOINFER_LOG_ERROR() \
    TURBOINFER_LOG(::turboinfer::util::LogLevel::kError)

#define TURBOINFER_LOG_FATAL() \
    TURBOINFER_LOG(::turboinfer::util::LogLevel::kFatal)

// Conditional logging macros
#define TURBOINFER_LOG_IF(level, condition) \
    if (condition) TURBOINFER_LOG(level)

#define TURBOINFER_LOG_DEBUG_IF(condition) \
    TURBOINFER_LOG_IF(::turboinfer::util::LogLevel::kDebug, condition)

#define TURBOINFER_LOG_INFO_IF(condition) \
    TURBOINFER_LOG_IF(::turboinfer::util::LogLevel::kInfo, condition)

#define TURBOINFER_LOG_WARNING_IF(condition) \
    TURBOINFER_LOG_IF(::turboinfer::util::LogLevel::kWarning, condition)

#define TURBOINFER_LOG_ERROR_IF(condition) \
    TURBOINFER_LOG_IF(::turboinfer::util::LogLevel::kError, condition)

#define TURBOINFER_LOG_FATAL_IF(condition) \
    TURBOINFER_LOG_IF(::turboinfer::util::LogLevel::kFatal, condition)
