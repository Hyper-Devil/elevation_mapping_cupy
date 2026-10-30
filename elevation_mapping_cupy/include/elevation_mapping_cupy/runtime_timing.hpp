#pragma once
// Optional deployment audit. No ROS/simulated clock is used for durations.
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <unistd.h>

namespace elevation_mapping_cupy {
class RuntimeTimingScope {
 public:
  static std::ofstream& output() {
    static std::ofstream file;
    static const bool opened = [] {
      const char* dir = std::getenv("UAM2TE_PROFILE_DIR");
      if (dir && *dir) {
        file.open(std::string(dir) + "/mapping_cpp.csv", std::ios::out);
        file << "stage,start_ns,end_ns,stamp_ns,pid\n";
      }
      return true;
    }();
    (void)opened;
    return file;
  }
  static long long now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  RuntimeTimingScope(const char* stage, unsigned long long stamp = 0)
      : enabled_(output().is_open()), stage_(stage), stamp_(stamp) {
    if (enabled_) start_ = last_ = now();
  }
  ~RuntimeTimingScope() { if (enabled_) record(stage_, start_, now()); }
  void mark(const char* stage) {
    if (!enabled_) return;
    const auto end = now();
    record(stage, last_, end);
    last_ = now();
  }
 private:
  void record(const char* stage, long long start, long long end) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    output() << stage << ',' << start << ',' << end << ',' << stamp_ << ',' << getpid() << '\n';
    output().flush();
  }
  bool enabled_;
  const char* stage_;
  unsigned long long stamp_;
  long long start_ = 0, last_ = 0;
};
}
