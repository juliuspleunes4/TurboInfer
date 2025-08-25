/**
 * @file logging.cpp
 * @brief Implementation of logging utilities (placeholder).
 * @author J.J.G. Pleunes
 */

#include "turboinfer/util/logging.hpp"
#include <iostream>
#include <fstream>
#include <mutex>
#include <iomanip>
#include <chrono>
#include <ctime>

namespace turboinfer {
namespace util {

// Logger implementation details
class LoggerImpl {
public:
    std::mutex mutex_;
    std::unique_ptr<std::ofstream> file_stream_;
};

// LogStream implementation

LogStream::LogStream(LogLevel level, const char* file, int line)
    : level_(level), file_(file), line_(line) {
}

LogStream::~LogStream() {
    Logger::instance().log(level_, stream_.str(), file_, line_);
}

// Logger implementation

Logger::Logger() : impl_(std::make_unique<LoggerImpl>()) {
}

Logger::~Logger() = default;

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

bool Logger::set_file_output(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    
    if (file_path.empty()) {
        impl_->file_stream_.reset();
        return true;
    }
    
    impl_->file_stream_ = std::make_unique<std::ofstream>(file_path, std::ios::app);
    return impl_->file_stream_->is_open();
}

void Logger::log(LogLevel level, const std::string& message, const char* file, int line) {
    if (level < level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    
    std::ostringstream log_line;
    
    // Add timestamp if enabled
    if (timestamp_enabled_) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        log_line << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        log_line << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
    }
    
    // Add log level
    log_line << "[" << log_level_to_string(level) << "] ";
    
    // Add message
    log_line << message;
    
    // Add file and line if provided
    if (file && file[0] != '\0' && line > 0) {
        // Extract just the filename from the full path
        const char* filename = file;
        const char* last_slash = nullptr;
        for (const char* p = file; *p; ++p) {
            if (*p == '/' || *p == '\\') {
                last_slash = p;
            }
        }
        if (last_slash) {
            filename = last_slash + 1;
        }
        
        log_line << " (" << filename << ":" << line << ")";
    }
    
    std::string final_message = log_line.str();
    
    // Output to console if enabled
    if (console_output_) {
        if (level >= LogLevel::kError) {
            std::cerr << final_message << std::endl;
        } else {
            std::cout << final_message << std::endl;
        }
    }
    
    // Output to file if configured
    if (impl_->file_stream_ && impl_->file_stream_->is_open()) {
        *impl_->file_stream_ << final_message << std::endl;
    }
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    
    if (console_output_) {
        std::cout.flush();
        std::cerr.flush();
    }
    
    if (impl_->file_stream_ && impl_->file_stream_->is_open()) {
        impl_->file_stream_->flush();
    }
}

// Utility functions

const char* log_level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::kDebug: return "DEBUG";
        case LogLevel::kInfo: return "INFO";
        case LogLevel::kWarning: return "WARN";
        case LogLevel::kError: return "ERROR";
        case LogLevel::kFatal: return "FATAL";
        default: return "UNKNOWN";
    }
}

LogLevel string_to_log_level(const std::string& str) {
    if (str == "DEBUG") return LogLevel::kDebug;
    if (str == "INFO") return LogLevel::kInfo;
    if (str == "WARN" || str == "WARNING") return LogLevel::kWarning;
    if (str == "ERROR") return LogLevel::kError;
    if (str == "FATAL") return LogLevel::kFatal;
    
    throw std::invalid_argument("Invalid log level string: " + str);
}

} // namespace util
} // namespace turboinfer
