#include "vp_frame_batch_analyser_node.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm_client.h"

namespace trashbot {

namespace {

constexpr double kFallbackFps = 25.0;

// Standard base64 alphabet with padding.
const char kBase64Chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const unsigned char* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned int v = static_cast<unsigned int>(data[i]) << 16;
        if (i + 1 < len) {
            v |= static_cast<unsigned int>(data[i + 1]) << 8;
        }
        if (i + 2 < len) {
            v |= static_cast<unsigned int>(data[i + 2]);
        }
        out += kBase64Chars[(v >> 18) & 0x3F];
        out += kBase64Chars[(v >> 12) & 0x3F];
        out += (i + 1 < len) ? kBase64Chars[(v >> 6) & 0x3F] : '=';
        out += (i + 2 < len) ? kBase64Chars[v & 0x3F] : '=';
    }
    return out;
}

// Format seconds as mm:ss.
std::string format_mmss(double seconds) {
    long t = static_cast<long>(std::llround(seconds));
    if (t < 0) {
        t = 0;
    }
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%02ld:%02ld", t / 60, t % 60);
    return buf;
}

// Draw OSD lines with a dark box behind them. The box keeps the text
// readable on bright footage. Font size follows the frame width, so 4K
// frames get legible text. Box height follows the line count.
void draw_osd_lines(cv::Mat& frame,
                    const std::vector<std::string>& lines) {
    if (lines.empty()) {
        return;
    }
    double scale = static_cast<double>(frame.cols) / 1600.0;
    if (scale < 0.7) {
        scale = 0.7;
    }
    int thickness = static_cast<int>(scale);
    if (thickness < 1) {
        thickness = 1;
    }
    int baseline = 0;
    int max_w = 0;
    for (const auto& l : lines) {
        const auto s = cv::getTextSize(l, cv::FONT_HERSHEY_SIMPLEX,
                                       scale, thickness, &baseline);
        if (s.width > max_w) {
            max_w = s.width;
        }
    }
    const auto single = cv::getTextSize(lines.front(),
                                        cv::FONT_HERSHEY_SIMPLEX,
                                        scale, thickness, &baseline);
    const int line_h = single.height + baseline + 6;
    const int pad = 8;
    const int block_h = line_h * static_cast<int>(lines.size());
    const int left = 10;
    // Bottom margin scales with the frame, so descenders keep clear of
    // the edge on high-resolution footage.
    const int bottom_margin = std::max(12, frame.rows / 48);
    const int bottom = frame.rows - bottom_margin;
    cv::rectangle(frame,
                  cv::Rect(left - pad, bottom - block_h - pad,
                           max_w + 2 * pad, block_h + 2 * pad),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    int y = bottom - block_h + line_h - baseline / 2;
    for (const auto& l : lines) {
        cv::putText(frame, l, cv::Point(left, y), cv::FONT_HERSHEY_SIMPLEX,
                    scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        y += line_h;
    }
}

std::string replace_all(std::string text, const std::string& from,
                        const std::string& to) {
    if (from.empty()) {
        return text;
    }
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

// Remove markdown fence markers. The prompt asks for plain JSON, but some
// models wrap the answer in fences anyway.
std::string strip_markdown_fences(const std::string& text) {
    std::string out = text;
    const std::string fence = "```";
    size_t pos = 0;
    while ((pos = out.find(fence, pos)) != std::string::npos) {
        out.erase(pos, fence.size());
    }
    return out;
}

// Find the first balanced JSON object in text. The scan respects strings.
// Return true and set out on success.
bool find_first_json_object(const std::string& text, std::string& out) {
    size_t search_pos = 0;
    while (search_pos < text.size()) {
        const size_t start = text.find('{', search_pos);
        if (start == std::string::npos) {
            return false;
        }
        int depth = 0;
        bool in_string = false;
        bool escaped = false;
        for (size_t i = start; i < text.size(); ++i) {
            const char c = text[i];
            if (in_string) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    in_string = false;
                }
                continue;
            }
            if (c == '"') {
                in_string = true;
            } else if (c == '{') {
                ++depth;
            } else if (c == '}') {
                --depth;
                if (depth == 0) {
                    std::string candidate = text.substr(start, i - start + 1);
                    try {
                        // Parse as validation only. Discard the result.
                        auto parsed = nlohmann::json::parse(candidate);
                        (void)parsed;
                        out = candidate;
                        return true;
                    } catch (const nlohmann::json::exception&) {
                        break;  // try the next opening brace
                    }
                }
            }
        }
        search_pos = start + 1;
    }
    return false;
}

}  // namespace

vp_frame_batch_analyser_node::vp_frame_batch_analyser_node(
        std::string node_name, const batch_analyser_params& params,
        const std::string& video_path)
    : vp_node(node_name), params_(params), video_path_(video_path) {
    if (params_.batch < 1) {
        params_.batch = 1;
    }
    // An empty params_.sample_id keeps the old behavior: the file stem.
    sample_id_ = params_.sample_id.empty()
        ? std::filesystem::path(video_path_).filename().stem().string()
        : params_.sample_id;

    jsonl_.open(params_.jsonl_path, std::ios::out | std::ios::trunc);
    if (!jsonl_.is_open()) {
        throw std::runtime_error(
            "cannot open jsonl output file: " + params_.jsonl_path);
    }

    // Start the internal threads. Call this at the end of the constructor.
    initialized();
}

vp_frame_batch_analyser_node::~vp_frame_batch_analyser_node() {
    // Stop the node threads before members die.
    deinitialized();
    // Threads joined. Write out what is left in the buffer.
    flush_window(nullptr);
}

void vp_frame_batch_analyser_node::set_max_windows(int max_windows) {
    max_windows_ = max_windows < 0 ? 0 : max_windows;
}

size_t vp_frame_batch_analyser_node::pending_samples() {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
}

size_t vp_frame_batch_analyser_node::pending_frames() {
    std::lock_guard<std::mutex> lock(in_queue_lock);
    return in_queue.size();
}

void vp_frame_batch_analyser_node::request_flush() {
    // Send a SPEAK control meta through the normal queue. The handle thread
    // serializes the flush with frame handling. No extra locks needed.
    auto ctrl = std::make_shared<vp_objects::vp_control_meta>(
        vp_objects::vp_control_type::SPEAK, channel_index_,
        "trashbot_flush");
    meta_flow(ctrl);
}

bool vp_frame_batch_analyser_node::limit_reached() const {
    return max_windows_ > 0
        && windows_flushed_.load(std::memory_order_relaxed)
               + windows_failed_.load(std::memory_order_relaxed)
           >= max_windows_;
}

std::shared_ptr<vp_objects::vp_meta> vp_frame_batch_analyser_node::handle_frame_meta(
        std::shared_ptr<vp_objects::vp_frame_meta> meta) {
    frames_seen_.fetch_add(1, std::memory_order_relaxed);

    // Drop frames once the caller hit the window limit.
    if (limit_reached()) {
        return nullptr;
    }

    // Compute the stride from the real stream fps at the first frame.
    // Take every round(source_fps / target_fps)-th frame.
    if (stride_ <= 0) {
        if (params_.source_fps > 0) {
            effective_fps_ = params_.source_fps;
        } else {
            effective_fps_ = meta->fps > 0 ? static_cast<double>(meta->fps)
                                           : kFallbackFps;
        }
        const double target = params_.fps > 0 ? params_.fps : 1.0;
        long s = static_cast<long>(std::llround(effective_fps_ / target));
        stride_ = s < 1 ? 1 : s;
    }

    if (meta->frame_index >= next_sample_index_) {
        next_sample_index_ = meta->frame_index + stride_;

        // Encode now. Keep only the small JPEG payload in memory.
        std::vector<uchar> buf;
        const std::vector<int> enc_params = {cv::IMWRITE_JPEG_QUALITY,
                                             params_.jpeg_quality};
        cv::imencode(".jpg", meta->frame, buf, enc_params);

        sampled_frame sf;
        sf.jpeg_base64 = base64_encode(buf.data(), buf.size());
        sf.frame_index = meta->frame_index;
        sf.timestamp_s = meta->frame_index / effective_fps_;

        bool full = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            buffer_.push_back(std::move(sf));
            full = static_cast<long>(buffer_.size()) >= params_.batch;
        }
        if (full && !limit_reached()) {
            flush_window(meta.get());
        }
    }

    // Pass every frame through. Stamp the latest window summary on it, so
    // overlay videos show annotations all the time, not on one frame.
    if (!last_osd_lines_.empty() && !meta->frame.empty()) {
        draw_osd_lines(meta->frame, last_osd_lines_);
    }
    return limit_reached() ? nullptr : meta;
}

std::shared_ptr<vp_objects::vp_meta> vp_frame_batch_analyser_node::handle_control_meta(
        std::shared_ptr<vp_objects::vp_control_meta> meta) {
    // EOF signal: main.cpp calls src->speak() when it detects end of stream.
    // vp_file_src_node never signals EOF downstream by itself, so this is
    // the clean way to flush a partial tail window through the pipe.
    if (meta->control_type == vp_objects::vp_control_type::SPEAK) {
        flush_window(nullptr);
    }
    return meta;
}

void vp_frame_batch_analyser_node::flush_window(
        vp_objects::vp_frame_meta* last_frame) {
    std::vector<sampled_frame> window;
    long window_idx = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.empty()) {
            return;
        }
        if (limit_reached()) {
            buffer_.clear();
            return;
        }
        window_idx = window_idx_++;
        window.swap(buffer_);
    }

    busy_.store(true, std::memory_order_relaxed);

    const double t_start = window.front().timestamp_s;
    const double t_end = window.back().timestamp_s;

    // Prompt with substituted placeholders, then one text part per frame
    // label and image part, then the closing instruction line.
    std::string prompt = params_.prompt_template_text;
    prompt = replace_all(prompt, "{N_FRAMES}", std::to_string(window.size()));
    prompt = replace_all(prompt, "{T_START}", format_mmss(t_start));
    prompt = replace_all(prompt, "{T_END}", format_mmss(t_end));

    nlohmann::ordered_json content = nlohmann::ordered_json::array();
    content.push_back({{"type", "text"}, {"text", prompt}});
    int k = 1;
    for (const auto& f : window) {
        content.push_back({{"type", "text"},
                           {"text", "Frame " + std::to_string(k)
                                        + " (t=" + format_mmss(f.timestamp_s)
                                        + "):"}});
        content.push_back(
            {{"type", "image_url"},
             {"image_url",
              {{"url", "data:image/jpeg;base64," + f.jpeg_base64}}}});
        ++k;
    }
    content.push_back({{"type", "text"},
                       {"text", "Answer with the JSON object now."}});

    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    messages.push_back({{"role", "user"}, {"content", content}});

    VllmParams vparams;
    vparams.endpoint_url = params_.endpoint_url;
    vparams.model = params_.model;
    vparams.api_key = params_.api_key;
    vparams.max_tokens = params_.max_tokens;
    vparams.temperature = params_.temperature;
    vparams.seed = params_.seed;
    vparams.timeout_s = params_.timeout_s;

    std::string response_raw;
    std::string error_msg;  // empty means no error
    nlohmann::json annotation;
    try {
        response_raw = analyse(vparams, messages);
        std::string obj_text;
        if (find_first_json_object(strip_markdown_fences(response_raw),
                                   obj_text)) {
            annotation = nlohmann::json::parse(obj_text);
        } else {
            error_msg = "no JSON object found in model answer";
        }
    } catch (const std::exception& e) {
        error_msg = e.what();
    }

    // One JSON line per window. ordered_json keeps the documented key order.
    nlohmann::ordered_json line;
    line["schema_version"] = 1;
    line["video_path"] = video_path_;
    line["sample_id"] = sample_id_;
    line["window_idx"] = window_idx;
    line["t_start_s"] = t_start;
    line["t_end_s"] = t_end;
    auto indices = nlohmann::ordered_json::array();
    auto stamps = nlohmann::ordered_json::array();
    for (const auto& f : window) {
        indices.push_back(f.frame_index);
        stamps.push_back(f.timestamp_s);
    }
    line["frame_indices"] = indices;
    line["timestamps_s"] = stamps;
    line["model"] = params_.model;
    line["response_raw"] = response_raw;
    line["annotation"] = error_msg.empty() ? annotation : nlohmann::json();
    line["error"] = error_msg.empty() ? nlohmann::json()
                                      : nlohmann::json(error_msg);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        jsonl_ << line.dump() << "\n";
        jsonl_.flush();
    }

    int interactions = 0;
    if (error_msg.empty() && annotation.is_object()
        && annotation.contains("interactions")
        && annotation["interactions"].is_array()) {
        interactions = static_cast<int>(annotation["interactions"].size());
    }

    // Build the persistent OSD lines: window header, interaction summary,
    // then the strongest norm dimensions. Keep the block short so it stays
    // readable on a moving frame.
    std::vector<std::string> osd_lines;
    osd_lines.push_back("TrashBot w" + std::to_string(window_idx)
                        + " [" + format_mmss(t_start) + "-"
                        + format_mmss(t_end) + "]");
    if (!error_msg.empty()) {
        osd_lines.push_back("error: " + error_msg.substr(0, 48));
    } else {
        std::vector<std::string> cats;
        if (annotation.is_object() && annotation.contains("interactions")
            && annotation["interactions"].is_array()) {
            for (const auto& it : annotation["interactions"]) {
                if (!it.contains("category")
                    || !it["category"].is_string()) {
                    continue;
                }
                const auto c = it["category"].get<std::string>();
                if (std::find(cats.begin(), cats.end(), c) == cats.end()) {
                    cats.push_back(c);
                }
            }
        }
        std::string line2 = "interactions=" + std::to_string(interactions);
        for (size_t i = 0; i < cats.size(); ++i) {
            line2 += i == 0 ? ": " : ", ";
            line2 += cats[i];
        }
        osd_lines.push_back(line2);

        // Top dimensions by relevance, strongest first. Truncate long
        // names at a word boundary; stars carry the score.
        std::vector<std::pair<int, std::string>> dims;
        if (annotation.is_object() && annotation.contains("norm_dimensions")
            && annotation["norm_dimensions"].is_array()) {
            for (const auto& d : annotation["norm_dimensions"]) {
                if (!d.contains("dimension")
                    || !d["dimension"].is_string()) {
                    continue;
                }
                if (!d.contains("relevance")
                    || !d["relevance"].is_number_integer()) {
                    continue;
                }
                const int rel = d["relevance"].get<int>();
                if (rel > 0) {
                    dims.emplace_back(rel, d["dimension"].get<std::string>());
                }
            }
        }
        std::stable_sort(dims.begin(), dims.end(),
                         [](const auto& a, const auto& b) {
                             return a.first > b.first;
                         });
        for (const auto& dim : dims) {
            if (osd_lines.size() >= 6) {
                break;
            }
            std::string name = dim.second;
            if (name.size() > 28) {
                name = name.substr(0, 28);
                const auto sp = name.rfind(' ');
                if (sp != std::string::npos && sp > 12) {
                    name = name.substr(0, sp);
                }
            }
            osd_lines.push_back(name + " " + std::string(dim.first, '*'));
        }
    }
    last_osd_lines_ = osd_lines;
    if (last_frame != nullptr && !last_frame->frame.empty()) {
        draw_osd_lines(last_frame->frame, osd_lines);
    }

    if (error_msg.empty()) {
        windows_flushed_.fetch_add(1, std::memory_order_relaxed);
    } else {
        windows_failed_.fetch_add(1, std::memory_order_relaxed);
    }
    busy_.store(false, std::memory_order_relaxed);
}

}  // namespace trashbot
