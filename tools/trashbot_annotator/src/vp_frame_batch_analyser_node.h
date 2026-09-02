#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <vp_control_meta.h>
#include <vp_frame_meta.h>
#include <vp_node.h>

namespace trashbot {

// Parameters for frame sampling and for the LLM call.
struct batch_analyser_params {
    std::string endpoint_url;
    std::string model;
    std::string api_key;
    std::string prompt_template_text;
    double fps = 1.0;            // target sample rate in frames per second
    // True rate of the frames which arrive at this node, in frames per
    // second. 0 means read the rate from the frame meta.
    //
    // Warning: vp_file_src_node keeps its fps in an int and divides it by
    // (skip_interval + 1). That division truncates to 0 as soon as the
    // skip interval is larger than the fps, and the meta then carries a
    // wrong rate. The caller knows the real rate, so it sets this field.
    double source_fps = 0.0;
    int batch = 8;               // sampled frames per window
    int jpeg_quality = 90;
    int max_tokens = 1024;
    double temperature = 0.0;
    long seed = 42;
    int timeout_s = 120;
    std::string jsonl_path;
    std::string sample_id;  // empty means derive from the video file name
};

// Node which samples frames, sends windows of JPEG frames to a VLM, and
// writes one JSON line per window to a jsonl file. All frames pass through
// unchanged. The latest window summary stays drawn as an OSD block on
// every passed frame, so overlay videos carry visible annotations
// throughout.
class vp_frame_batch_analyser_node : public vp_nodes::vp_node {
public:
    // video_path feeds the video_path and sample_id fields of the jsonl lines.
    vp_frame_batch_analyser_node(std::string node_name,
                                 const batch_analyser_params& params,
                                 const std::string& video_path);
    ~vp_frame_batch_analyser_node() override;

    // Stop after this many flushed windows. 0 means no limit.
    void set_max_windows(int max_windows);

    // Flush a partial window through the queue. Safe from any thread.
    void request_flush();

    long frames_seen() const { return frames_seen_.load(std::memory_order_relaxed); }
    long windows_flushed() const { return windows_flushed_.load(std::memory_order_relaxed); }
    long windows_failed() const { return windows_failed_.load(std::memory_order_relaxed); }
    bool busy() const { return busy_.load(std::memory_order_relaxed); }
    size_t pending_samples();

    // Count the frames which wait in the input queue. The vendor pops
    // without the lock, so read this as a close estimate. It is good
    // enough for backpressure.
    size_t pending_frames();

protected:
    std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(
        std::shared_ptr<vp_objects::vp_frame_meta> meta) override;

    // A SPEAK control meta triggers a flush of a partial window.
    // main.cpp sends it through src->speak() at end of stream.
    std::shared_ptr<vp_objects::vp_meta> handle_control_meta(
        std::shared_ptr<vp_objects::vp_control_meta> meta) override;

private:
    struct sampled_frame {
        std::string jpeg_base64;
        long frame_index = 0;
        double timestamp_s = 0.0;
    };

    // Runs on the handle thread only. last_frame is the last frame of the
    // window when the flush comes from frame handling, else null. The
    // drawer stamps the OSD on it, so the point is not const.
    void flush_window(vp_objects::vp_frame_meta* last_frame);

    bool limit_reached() const;

    batch_analyser_params params_;
    std::string video_path_;
    std::string sample_id_;
    int channel_index_ = 0;

    std::ofstream jsonl_;
    std::mutex mutex_;                 // guards buffer_, window_idx_
    std::vector<sampled_frame> buffer_;
    long window_idx_ = 0;

    // Sampling state. Set on the handle thread, read there only.
    long stride_ = 0;                  // 0 means not computed yet
    double effective_fps_ = 25.0;
    long next_sample_index_ = 0;

    int max_windows_ = 0;              // 0 means no limit

    // Latest window summary lines for the OSD. Written at flush time, read
    // at pass-through time. Both run on the handle thread.
    std::vector<std::string> last_osd_lines_;

    std::atomic<long> frames_seen_{0};
    std::atomic<long> windows_flushed_{0};
    std::atomic<long> windows_failed_{0};
    std::atomic<bool> busy_{false};
};

}  // namespace trashbot
