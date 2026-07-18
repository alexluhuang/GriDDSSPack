// Emacs Mode Line: -*- Mode:c++;-*-
// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   ca_async_writer.hpp
 * @brief  Phase-5 I/O: overlap CSV row formatting/writing with compute.
 *
 * The contingency loop hands each converged case's already-formatted CSV block
 * to this writer.  In async mode a single background thread (a spare Grace core)
 * drains a FIFO queue and appends blocks to the per-rank .part file, so disk
 * writing overlaps the next contingency's solve -- the Phase-5 lever that keeps
 * I/O off the critical path once compute is fast (gpuCA spent 145 s of 181 s on
 * CSV).  Because there is exactly ONE consumer draining a FIFO, and the loop
 * enqueues blocks in the same order it previously wrote them, the bytes are
 * IDENTICAL to the synchronous path -- so the three-CSV contract is preserved
 * and validated by the same compare_ca_csv.py oracle.
 *
 * Synchronous mode (the default) writes straight through and is byte-for-byte
 * the pre-existing behavior; async mode is opt-in via <overlapIO>true.
 * Header-only, standard C++11 threads only.
 */
// -------------------------------------------------------------

#ifndef _ca_async_writer_hpp_
#define _ca_async_writer_hpp_

#include <string>
#include <fstream>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace gridpack {
namespace contingency_analysis {

class AsyncRowWriter
{
public:

  AsyncRowWriter(void)
    : p_async(false), p_started(false), p_stop(false), p_wrote(false)
  {}

  ~AsyncRowWriter(void)
  {
    close();
  }

  /// Bind to a path.  @c async selects the background-writer (overlap) mode.
  /// The file is created lazily on the first non-empty block, so ranks that
  /// emit nothing leave no file behind (matching the prior behavior).
  void open(const std::string& path, bool async)
  {
    p_path = path;
    p_async = async;
    if (p_async) {
      p_stop = false;
      p_thread = std::thread(&AsyncRowWriter::p_run, this);
      p_started = true;
    }
  }

  /// Append a fully-formatted CSV block (e.g. all rows of one contingency).
  void write(const std::string& block)
  {
    if (block.empty()) return;
    if (!p_async) {
      p_ensureFile();
      p_file << block;
      p_wrote = true;
      return;
    }
    {
      std::lock_guard<std::mutex> lk(p_mtx);
      p_queue.push_back(block);
    }
    p_cv.notify_one();
  }

  /// Flush the queue, join the writer thread, and close the file.
  void close(void)
  {
    if (p_async && p_started) {
      {
        std::lock_guard<std::mutex> lk(p_mtx);
        p_stop = true;
      }
      p_cv.notify_one();
      if (p_thread.joinable()) p_thread.join();
      p_started = false;
    }
    if (p_file.is_open()) p_file.close();
  }

  /// True iff at least one non-empty block reached disk.
  bool wroteAnything(void) const { return p_wrote; }

private:

  void p_ensureFile(void)
  {
    if (!p_file.is_open()) {
      p_file.open(p_path.c_str(), std::ios::out | std::ios::trunc);
    }
  }

  void p_run(void)
  {
    for (;;) {
      std::deque<std::string> local;
      {
        std::unique_lock<std::mutex> lk(p_mtx);
        p_cv.wait(lk, [this] { return p_stop || !p_queue.empty(); });
        local.swap(p_queue);
      }
      for (std::size_t i = 0; i < local.size(); ++i) {
        if (!local[i].empty()) { p_ensureFile(); p_file << local[i]; p_wrote = true; }
      }
      // Exit only after draining everything the producer enqueued.
      {
        std::unique_lock<std::mutex> lk(p_mtx);
        if (p_stop && p_queue.empty()) break;
      }
    }
  }

  std::string p_path;
  bool p_async;
  bool p_started;
  bool p_stop;
  bool p_wrote;
  std::ofstream p_file;
  std::deque<std::string> p_queue;
  std::thread p_thread;
  std::mutex p_mtx;
  std::condition_variable p_cv;

  AsyncRowWriter(const AsyncRowWriter&);
  AsyncRowWriter& operator=(const AsyncRowWriter&);
};

} // namespace contingency_analysis
} // namespace gridpack

#endif
