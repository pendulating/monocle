// TrashBot deployment footage annotator.
//
// Pipeline: vp_file_src_node -> vp_frame_batch_analyser_node -> sink.
// The sink is vp_file_des_node with --overlay, else vp_fake_des_node.
#include <getopt.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <opencv2/videoio.hpp>
#include <nlohmann/json.hpp>

#include <vp_fake_des_node.h>
#include <vp_file_des_node.h>
#include <vp_file_src_node.h>
#include <logger/vp_logger.h>

#include "vp_frame_batch_analyser_node.h"

namespace {

constexpr int kChannelIndex = 0;

// Cap for the analyser input queue. A 720p frame costs about 2.8 MB, so
// this cap holds about 130 MB. TRASHBOT_MAX_PENDING_FRAMES changes it.
constexpr long kDefaultMaxPendingFrames = 48;

long max_pending_frames() {
    const char* raw = std::getenv("TRASHBOT_MAX_PENDING_FRAMES");
    if (raw == nullptr || raw[0] == '\0') {
        return kDefaultMaxPendingFrames;
    }
    const long value = std::strtol(raw, nullptr, 10);
    return value > 0 ? value : kDefaultMaxPendingFrames;
}

// VideoPipe nodes log through a global logger. Start it before any node
// exists.
//
// Warning: the vendor default is DEBUG level to the console and to a file.
// That writes some lines per frame. One long video made a 3 GB log file and
// a stdout stream of the same size, which filled /tmp and grew the parent
// process that captures stdout. Keep the level at WARN and the console off.
// TRASHBOT_LOG_DIR moves the log file; TRASHBOT_LOG_LEVEL raises the level
// again for a debug session.
void init_vp_logger() {
    auto& logger = vp_utils::vp_logger::get_logger();

    const char* dir = std::getenv("TRASHBOT_LOG_DIR");
    logger.log_dir = (dir != nullptr && dir[0] != '\0') ? dir : "./logs/vp";

    logger.log_level = vp_utils::vp_log_level::WARN;
    const char* level = std::getenv("TRASHBOT_LOG_LEVEL");
    if (level != nullptr) {
        const std::string want(level);
        if (want == "ERROR") {
            logger.log_level = vp_utils::vp_log_level::ERROR;
        } else if (want == "INFO") {
            logger.log_level = vp_utils::vp_log_level::INFO;
        } else if (want == "DEBUG") {
            logger.log_level = vp_utils::vp_log_level::DEBUG;
        }
    }

    // The driver captures stdout in memory. Console logs must stay small.
    logger.log_to_console = false;

    std::error_code ec;
    std::filesystem::create_directories(logger.log_dir, ec);
    logger.init();
}

struct cli_args {
    std::string video;
    std::string config = "./config/default.json";
    std::string jsonl;
    std::string overlay;
    std::string sample_id; // empty means derive from the video file name
    double fps = -1.0;   // negative means use the config value
    int batch = -1;      // negative means use the config value
    int max_windows = 0; // 0 means no limit
};

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " --video PATH --jsonl PATH [options]\n"
        << "  --video PATH        input video file (required)\n"
        << "  --jsonl PATH        output jsonl file (required)\n"
        << "  --config PATH       config file, default ./config/default.json\n"
        << "  --overlay PATH      write an annotated mp4 next to this path\n"
        << "  --sample-id STR     sample id for jsonl records, default is the video file stem\n"
        << "  --fps F             target sample rate, overrides config\n"
        << "  --batch N           frames per window, overrides config\n"
        << "  --max-windows N     stop after N flushed windows, 0 means no limit\n"
        << "  --help              show this help\n";
}

bool parse_cli(int argc, char** argv, cli_args& args) {
    static struct option long_opts[] = {
        {"video", required_argument, nullptr, 'v'},
        {"config", required_argument, nullptr, 'c'},
        {"jsonl", required_argument, nullptr, 'j'},
        {"overlay", required_argument, nullptr, 'o'},
        {"sample-id", required_argument, nullptr, 's'},
        {"fps", required_argument, nullptr, 'f'},
        {"batch", required_argument, nullptr, 'b'},
        {"max-windows", required_argument, nullptr, 'w'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "v:c:j:o:s:f:b:w:h", long_opts,
                              nullptr)) != -1) {
        switch (opt) {
            case 'v': args.video = optarg; break;
            case 'c': args.config = optarg; break;
            case 'j': args.jsonl = optarg; break;
            case 'o': args.overlay = optarg; break;
            case 's': args.sample_id = optarg; break;
            case 'f':
                args.fps = std::atof(optarg);
                if (!(args.fps > 0)) {
                    std::cerr << "--fps needs a positive number, got: "
                              << optarg << "\n";
                    return false;
                }
                break;
            case 'b':
                args.batch = std::atoi(optarg);
                if (args.batch < 1) {
                    std::cerr << "--batch needs a positive integer, got: "
                              << optarg << "\n";
                    return false;
                }
                break;
            case 'w':
                args.max_windows = std::atoi(optarg);
                if (args.max_windows < 0) {
                    std::cerr << "--max-windows needs a non-negative integer, got: "
                              << optarg << "\n";
                    return false;
                }
                break;
            case 'h': print_usage(argv[0]); std::exit(0);
            default:
                print_usage(argv[0]);
                return false;
        }
    }
    if (args.video.empty()) {
        std::cerr << "--video is required\n";
        return false;
    }
    if (args.jsonl.empty()) {
        std::cerr << "--jsonl is required\n";
        return false;
    }
    return true;
}

trashbot::batch_analyser_params load_params(
        const std::string& config_path, const cli_args& args,
        std::string& prompt_text) {
    std::ifstream in(config_path);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open config file: " + config_path);
    }
    nlohmann::json cfg;
    try {
        in >> cfg;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("bad JSON in config file "
                                 + config_path + ": " + e.what());
    }

    // Fail fast on missing or wrongly typed keys.
    static const std::pair<std::string, bool> required[] = {
        {"endpoint_url", true},   {"model", true},          {"api_key", true},
        {"fps", false},           {"batch", false},         {"jpeg_quality", false},
        {"max_tokens", false},    {"temperature", false},   {"seed", false},
        {"timeout_s", false},     {"prompt_template", true},
    };
    for (const auto& key : required) {
        if (!cfg.contains(key.first)) {
            throw std::runtime_error("config misses key: " + key.first);
        }
    }

    trashbot::batch_analyser_params p;
    p.endpoint_url = cfg.at("endpoint_url").get<std::string>();
    p.model = cfg.at("model").get<std::string>();
    p.api_key = cfg.at("api_key").get<std::string>();
    p.fps = cfg.at("fps").get<double>();
    p.batch = cfg.at("batch").get<int>();
    p.jpeg_quality = cfg.at("jpeg_quality").get<int>();
    p.max_tokens = cfg.at("max_tokens").get<int>();
    p.temperature = cfg.at("temperature").get<double>();
    p.seed = cfg.at("seed").get<long>();
    p.timeout_s = cfg.at("timeout_s").get<int>();
    p.jsonl_path = args.jsonl;
    p.sample_id = args.sample_id;

    // CLI overrides beat the config values.
    if (args.fps > 0) {
        p.fps = args.fps;
    }
    if (args.batch > 0) {
        p.batch = args.batch;
    }

    // Read the prompt template now, fail fast when it is missing.
    const std::string template_path =
        cfg.at("prompt_template").get<std::string>();
    std::ifstream tin(template_path);
    if (!tin.is_open()) {
        throw std::runtime_error("cannot read prompt template file: "
                                 + template_path);
    }
    std::ostringstream ss;
    ss << tin.rdbuf();
    prompt_text = ss.str();
    if (prompt_text.empty()) {
        throw std::runtime_error("prompt template file is empty: "
                                 + template_path);
    }
    p.prompt_template_text = prompt_text;
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    init_vp_logger();

    cli_args args;
    if (!parse_cli(argc, argv, args)) {
        return 2;
    }

    std::string prompt_text;
    trashbot::batch_analyser_params params;
    try {
        params = load_params(args.config, args, prompt_text);
    } catch (const std::exception& e) {
        std::cerr << "config error: " << e.what() << "\n";
        return 2;
    }

    // Probe the video for the frame count. main.cpp needs it to detect the
    // end of stream, because vp_file_src_node never signals EOF downstream.
    cv::VideoCapture probe(args.video);
    if (!probe.isOpened()) {
        std::cerr << "cannot open video file: " << args.video << "\n";
        return 2;
    }
    const double total_frames_d = probe.get(cv::CAP_PROP_FRAME_COUNT);
    const double src_fps_d = probe.get(cv::CAP_PROP_FPS);
    probe.release();
    long total_frames =
        (total_frames_d > 0 && std::isfinite(total_frames_d))
            ? static_cast<long>(total_frames_d) : -1;

    // The source node pushes decoded frames into an unbounded queue at the
    // real-time rate. The analyser blocks on each HTTP call, so it reads
    // much slower than the source writes. On a long video the queue grows
    // until the job runs out of memory.
    //
    // The analyser keeps 1 frame in params.fps and drops the rest, so let
    // the source skip them first. This cuts the queued frames by the same
    // factor. The overlay sink needs every frame, so skip only when the
    // caller asks for no overlay.
    int skip_interval = 0;
    if (args.overlay.empty() && params.fps > 0 && src_fps_d > 0
        && std::isfinite(src_fps_d)) {
        const long step = std::lround(src_fps_d / params.fps);
        if (step > 1) {
            skip_interval = static_cast<int>(step - 1);
        }
    }
    // Tell the analyser the true arrival rate. It must not read the rate
    // from the frame meta, because the source truncates it to an int.
    if (src_fps_d > 0 && std::isfinite(src_fps_d)) {
        params.source_fps = src_fps_d / (skip_interval + 1);
    }

    std::cerr << "source fps=" << src_fps_d << " target fps=" << params.fps
              << " skip_interval=" << skip_interval
              << " effective fps=" << params.source_fps
              << " max_pending_frames=" << max_pending_frames() << "\n";
    if (skip_interval > 0 && total_frames > 0) {
        // frames_seen() counts the frames the source emits, not the frames
        // the file holds. The end-of-stream test compares against it.
        total_frames = total_frames / (skip_interval + 1);
    }

    // The backpressure hook below reads this flag. Declare it before the
    // nodes, so it outlives the hook which captures it.
    std::atomic<bool> producer_waits{false};

    // Declare order fixes destruction order: src dies first and stops the
    // meta flow, then the analyser joins its threads, then the sink dies.
    std::shared_ptr<vp_nodes::vp_node> des;
    if (!args.overlay.empty()) {
        // vp_file_des_node builds its own output name:
        // save_dir/name_prefix_<stamp>.mp4. We map --overlay PATH onto
        // save_dir = parent dir of PATH and name_prefix = stem of PATH.
        const std::filesystem::path out_path(args.overlay);
        auto save_dir = out_path.parent_path();
        if (save_dir.empty()) {
            save_dir = ".";
        }
        std::error_code ec;
        std::filesystem::create_directories(save_dir, ec);
        if (ec) {
            std::cerr << "cannot create overlay directory: "
                      << save_dir.string() << ": " << ec.message() << "\n";
            return 2;
        }
        // Cap the split duration at the assert limit of 30 minutes.
        des = std::make_shared<vp_nodes::vp_file_des_node>(
            "file_des_0", kChannelIndex, save_dir.string(),
            out_path.stem().string(), /*max_duration_for_single_file=*/30);
    } else {
        des = std::make_shared<vp_nodes::vp_fake_des_node>(
            "fake_des_0", kChannelIndex);
    }

    std::shared_ptr<trashbot::vp_frame_batch_analyser_node> analyser;
    std::shared_ptr<vp_nodes::vp_file_src_node> src;
    try {
        analyser = std::make_shared<trashbot::vp_frame_batch_analyser_node>(
            "batch_analyser_0", params, args.video);
        src = std::make_shared<vp_nodes::vp_file_src_node>(
            "file_src_0", kChannelIndex, args.video,
            /*resize_ratio=*/1.0f, /*cycle=*/false,
            /*gst_decoder_name=*/"avdec_h264", skip_interval);
    } catch (const std::exception& e) {
        std::cerr << "pipeline setup error: " << e.what() << "\n";
        return 2;
    }
    analyser->set_max_windows(args.max_windows);

    analyser->attach_to({src});
    des->attach_to({analyser});

    // Backpressure. The source pushes decoded frames into an unbounded
    // queue at the real-time rate. The analyser blocks on each HTTP call,
    // so it reads slower than the source writes. Without this hook the
    // queue grows until the job runs out of memory, which killed every
    // video longer than about 20 minutes.
    //
    // The vendor says a hook must not block. That rule protects a live
    // source, which loses frames when a reader stalls. Our source is a
    // file, so a slower read costs nothing. The source calls the hook on
    // its own produce thread and holds no lock there, so the wait is safe.
    const long cap = max_pending_frames();
    const long resume_at = cap / 2;
    const auto wait_limit = std::chrono::seconds(params.timeout_s + 120);
    src->set_meta_handled_hooker(
        [&analyser, &producer_waits, cap, resume_at, wait_limit](
            std::string, int, std::shared_ptr<vp_objects::vp_meta>) {
            if (static_cast<long>(analyser->pending_frames()) <= cap) {
                return;
            }
            const auto wait_start = std::chrono::steady_clock::now();
            producer_waits.store(true, std::memory_order_relaxed);
            while (static_cast<long>(analyser->pending_frames()) > resume_at) {
                // Safety net. A stuck analyser must not hang the source.
                if (std::chrono::steady_clock::now() - wait_start > wait_limit) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            producer_waits.store(false, std::memory_order_relaxed);
        });

    const auto t_start = std::chrono::steady_clock::now();
    src->start();

    // Monitor loop. Detect end of stream, then flush the tail window.
    bool eof_sent = false;
    bool done = false;
    long last_seen = -1;
    auto last_progress = std::chrono::steady_clock::now();
    auto eof_time = last_progress;
    auto last_speak = last_progress;
    while (!done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto now = std::chrono::steady_clock::now();
        const long seen = analyser->frames_seen();
        if (seen != last_seen) {
            last_seen = seen;
            last_progress = now;
        }
        const long windows_done =
            analyser->windows_flushed() + analyser->windows_failed();

        if (args.max_windows > 0 && windows_done >= args.max_windows) {
            break;
        }

        if (!eof_sent) {
            bool at_eof = false;
            if (total_frames > 0 && seen >= total_frames) {
                at_eof = true;
            } else if (seen > 0 && !analyser->busy()
                       && !producer_waits.load(std::memory_order_relaxed)
                       && now - last_progress > std::chrono::seconds(5)) {
                // No new frame arrives and no HTTP call runs. The stream
                // ended. This also covers a wrong probe frame count, which
                // the skip_interval division can make too high.
                at_eof = true;
            }
            if (at_eof) {
                eof_sent = true;
                eof_time = now;
                last_speak = now;
                src->speak();  // SPEAK control meta triggers the tail flush
            }
        } else {
            // Straggler frames can arrive after the first flush. Speak again
            // until the buffer stays empty.
            if (!analyser->busy() && analyser->pending_samples() > 0
                && now - last_speak > std::chrono::seconds(1)) {
                last_speak = now;
                src->speak();
            }
            if (!analyser->busy() && analyser->pending_samples() == 0) {
                done = true;
            }
            // Safety net against a stuck HTTP call.
            if (now - eof_time
                > std::chrono::seconds(params.timeout_s + 60)) {
                std::cerr << "warning: timeout on tail flush, stop anyway\n";
                done = true;
            }
        }
    }

    src->stop();

    // Finalize the overlay container explicitly. Node teardown does not
    // guarantee writer release, and an open mp4 holds back its moov atom.
    // Give queued frames a moment to drain first; the release is mutex
    // guarded, so a straggler write stays safe either way.
    if (des != nullptr) {
        auto* file_sink = dynamic_cast<vp_nodes::vp_file_des_node*>(des.get());
        if (file_sink != nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            file_sink->release_writer();
        }
    }

    const auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now() - t_start)
                               .count();
    std::cout << "windows=" << analyser->windows_flushed()
              << " errors=" << analyser->windows_failed()
              << " elapsed=" << elapsed_s << "s" << std::endl;
    return 0;
}
