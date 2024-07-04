//
// Created by juelin on 6/30/24.
//

#ifndef PICKLE_TIMER_HPP
#define PICKLE_TIMER_HPP
#include <chrono>

namespace pickle {
class Timer {
public:
  // Starts the timer
  void start() { start_time = std::chrono::high_resolution_clock::now(); }

  // Stops the timer
  void end() { end_time = std::chrono::high_resolution_clock::now(); }

  // Calculates and returns the duration between start and end in milliseconds
  double microseconds() const {
    auto elapsed = end_time - start_time;
    return std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
        .count();
  }

  // Calculates and returns the duration between start and end in milliseconds
  double seconds() const { return microseconds() / 1e6; }

  // Resets the timer (sets start and end times to the current time)
  void reset() {
    start_time = end_time = std::chrono::high_resolution_clock::now();
  }

private:
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
};
} // namespace pickle
#endif // PICKLE_TIMER_HPP
