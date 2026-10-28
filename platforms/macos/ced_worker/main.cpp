#include "audio_io.hpp"
#include "ced.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
constexpr int kParentLost = 70;
constexpr int kProtocolVersion = 1;

struct Options {
    std::string model, input, output, expected_build, requested_device;
    double window_seconds = 0, hop_seconds = 0, stream_start_seconds = 0, source_duration_seconds = 0;
    long max_windows = 0, max_output_bytes = 0, expected_sample_rate = 0, parent_pid = 0;
};

bool finite_positive(double value) { return std::isfinite(value) && value > 0; }
bool parent_alive(long expected) { return expected > 0 && getppid() == expected; }

long exact_sample_count(double seconds, long sample_rate, const char* option) {
    const double samples = seconds * sample_rate;
    const long rounded = std::lround(samples);
    if (
        rounded <= 0
        || std::fabs(samples - static_cast<double>(rounded)) > 1e-6
    ) {
        throw std::runtime_error(
            std::string(option) + " must be an exact PCM sample duration"
        );
    }
    return rounded;
}

bool uses_requested_device(const std::string& observed, const std::string& requested) {
    if (requested == "metal") {
        return observed.size() >= 3
            && std::toupper(static_cast<unsigned char>(observed[0])) == 'M'
            && std::toupper(static_cast<unsigned char>(observed[1])) == 'T'
            && std::toupper(static_cast<unsigned char>(observed[2])) == 'L';
    }
    if (requested == "cpu") {
        return observed.size() == 3
            && std::tolower(static_cast<unsigned char>(observed[0])) == 'c'
            && std::tolower(static_cast<unsigned char>(observed[1])) == 'p'
            && std::tolower(static_cast<unsigned char>(observed[2])) == 'u';
    }
    return false;
}

std::string escape_json(const std::string& input) {
    std::ostringstream output;
    for (unsigned char c : input) {
        switch (c) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (c < 0x20) output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
            else output << c;
        }
    }
    return output.str();
}

bool regular_file(const std::string& name) {
    struct stat value {};
    return lstat(name.c_str(), &value) == 0 && S_ISREG(value.st_mode) && !S_ISLNK(value.st_mode);
}

std::string sha256_file(const std::string& name) {
    std::ifstream input(name, std::ios::binary);
    if (!input) return {};
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    char buffer[1024 * 1024];
    while (input.good()) {
        input.read(buffer, sizeof(buffer));
        if (input.gcount() > 0) CC_SHA256_Update(&context, buffer, static_cast<CC_LONG>(input.gcount()));
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    std::ostringstream output;
    for (unsigned char item : digest) output << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(item);
    return output.str();
}

bool take(const char* name, int& index, int argc, char** argv, std::string& target) {
    if (std::strcmp(argv[index], name)) return false;
    if (++index >= argc) throw std::runtime_error(std::string("missing value for ") + name);
    target = argv[index];
    return true;
}
template <typename T>
bool take_number(const char* name, int& index, int argc, char** argv, T& target) {
    if (std::strcmp(argv[index], name)) return false;
    if (++index >= argc) throw std::runtime_error(std::string("missing value for ") + name);
    char* end = nullptr;
    if constexpr (std::is_same<T, double>::value) target = std::strtod(argv[index], &end);
    else target = std::strtol(argv[index], &end, 10);
    if (!end || *end) throw std::runtime_error(std::string("invalid value for ") + name);
    return true;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (take("--model", i, argc, argv, options.model) || take("--input", i, argc, argv, options.input) ||
            take("--output", i, argc, argv, options.output) || take("--expected-build-id", i, argc, argv, options.expected_build) ||
            take("--requested-device", i, argc, argv, options.requested_device) ||
            take_number("--window-seconds", i, argc, argv, options.window_seconds) ||
            take_number("--hop-seconds", i, argc, argv, options.hop_seconds) ||
            take_number("--max-windows", i, argc, argv, options.max_windows) ||
            take_number("--max-output-bytes", i, argc, argv, options.max_output_bytes) ||
            take_number("--expected-sample-rate-hz", i, argc, argv, options.expected_sample_rate) ||
            take_number("--audio-stream-start-seconds", i, argc, argv, options.stream_start_seconds) ||
            take_number("--source-duration-seconds", i, argc, argv, options.source_duration_seconds) ||
            take_number("--parent-pid", i, argc, argv, options.parent_pid)) continue;
        throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
    }
    if (options.model.empty() || options.input.empty() || options.output.empty() || options.expected_build.empty() || options.requested_device.empty() ||
        !finite_positive(options.window_seconds) || !finite_positive(options.hop_seconds) || options.hop_seconds > options.window_seconds ||
        options.max_windows <= 0 || options.max_output_bytes <= 0 || options.expected_sample_rate <= 0 || options.parent_pid <= 0 ||
        !std::isfinite(options.stream_start_seconds) || !finite_positive(options.source_duration_seconds)
        || (options.requested_device != "metal" && options.requested_device != "cpu")) throw std::runtime_error("invalid CED worker arguments");
    if (options.expected_build != OPENSHORTS_CED_CPP_REVISION) throw std::runtime_error("expected build identifier does not match this worker");
    if (!regular_file(options.model) || !regular_file(options.input)) throw std::runtime_error("model and input must be regular files");
    return options;
}

void version() {
    std::cout << "{\"protocol_version\":" << kProtocolVersion
              << ",\"worker_build\":\"" << escape_json(OPENSHORTS_CED_CPP_REVISION)
              << "\",\"metal_compiled\":" << (OPENSHORTS_CED_METAL ? "true" : "false") << "}\n";
}

int probe(int argc, char** argv) {
    Options options;
    for (int i = 2; i < argc; ++i) {
        if (take("--model", i, argc, argv, options.model) ||
            take("--expected-build-id", i, argc, argv, options.expected_build) ||
            take("--requested-device", i, argc, argv, options.requested_device) ||
            take_number("--parent-pid", i, argc, argv, options.parent_pid)) continue;
        throw std::runtime_error(std::string("unknown probe argument: ") + argv[i]);
    }
    if (
        options.model.empty() || options.expected_build.empty() ||
        options.requested_device.empty() || options.parent_pid <= 0
    ) throw std::runtime_error("invalid CED worker probe arguments");
    if (options.expected_build != OPENSHORTS_CED_CPP_REVISION) {
        throw std::runtime_error("expected build identifier does not match this worker");
    }
    if (!parent_alive(options.parent_pid) || !regular_file(options.model)) return kParentLost;
    const std::string model_sha = sha256_file(options.model);
    if (model_sha.empty()) throw std::runtime_error("cannot hash model");
    ced::Ced model;
    if (!model.load(options.model)) {
        throw std::runtime_error("CED model load failed: " + model.load_error());
    }
    if (!uses_requested_device(model.device_name(), options.requested_device)) {
        throw std::runtime_error("CED worker could not initialize the requested device");
    }
    if (!parent_alive(options.parent_pid)) return kParentLost;
    std::cout << "{\"protocol_version\":" << kProtocolVersion
              << ",\"worker_build\":\"" << escape_json(OPENSHORTS_CED_CPP_REVISION)
              << "\",\"model_sha256\":\"" << model_sha << "\",\"runtime\":{\"requested_device\":\""
              << escape_json(options.requested_device) << "\",\"observed_device\":\""
              << escape_json(model.device_name()) << "\"}}\n";
    return 0;
}

bool write_result(const Options& options, const ced::Ced& model, const ced::Audio& audio, const std::string& model_sha, const std::vector<std::pair<double, std::vector<float>>>& outputs, const std::vector<double>& ends, const std::vector<double>& padded) {
    std::ostringstream json;
    json << std::setprecision(10);
    json << "{\"schema_version\":1,\"provider\":\"ced\",\"worker_build\":\"" << escape_json(OPENSHORTS_CED_CPP_REVISION)
         << "\",\"model_sha256\":\"" << model_sha << "\",\"runtime\":{\"requested_device\":\"" << escape_json(options.requested_device)
         << "\",\"observed_device\":\"" << escape_json(model.device_name()) << "\"},\"audio\":{\"sample_rate_hz\":" << audio.sample_rate
         << ",\"stream_start_seconds\":" << options.stream_start_seconds << ",\"window_seconds\":" << options.window_seconds
         << ",\"hop_seconds\":" << options.hop_seconds << ",\"decoded_duration_seconds\":"
         << static_cast<double>(audio.samples.size()) / audio.sample_rate
         << ",\"window_count\":" << outputs.size() << "},\"labels\":[";
    const auto& labels = model.config().labels;
    for (size_t i = 0; i < labels.size(); ++i) { if (i) json << ','; json << '"' << escape_json(labels[i]) << '"'; }
    json << "],\"windows\":[";
    for (size_t window = 0; window < outputs.size(); ++window) {
        if (window) json << ',';
        json << "{\"start_seconds\":" << outputs[window].first << ",\"end_seconds\":" << ends[window]
             << ",\"padded_tail_seconds\":" << padded[window] << ",\"scores\":[";
        const auto& scores = outputs[window].second;
        for (size_t i = 0; i < scores.size(); ++i) { if (i) json << ','; json << scores[i]; }
        json << "]}";
    }
    json << "]}";
    const std::string body = json.str();
    if (body.size() > static_cast<size_t>(options.max_output_bytes)) throw std::runtime_error("result exceeds configured output size limit");
    if (!parent_alive(options.parent_pid)) return false;
    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open worker output");
    output.write(body.data(), static_cast<std::streamsize>(body.size()));
    output.flush();
    if (!output) throw std::runtime_error("cannot write worker output");
    return true;
}

int run(int argc, char** argv) {
    Options options = parse(argc, argv);
    if (!parent_alive(options.parent_pid)) return kParentLost;
    std::string model_sha = sha256_file(options.model);
    if (model_sha.empty()) throw std::runtime_error("cannot hash model");
    ced::Ced model;
    if (!model.load(options.model)) throw std::runtime_error("CED model load failed: " + model.load_error());
    if (!uses_requested_device(model.device_name(), options.requested_device)) {
        throw std::runtime_error("CED worker could not initialize the requested device");
    }
    if (!parent_alive(options.parent_pid)) return kParentLost;
    if (model.config().sample_rate != options.expected_sample_rate) throw std::runtime_error("model sample rate differs from sealed profile");
    ced::Audio audio;
    if (!ced::load_audio_mono(options.input, audio, static_cast<int>(options.expected_sample_rate))) throw std::runtime_error("cannot read worker WAV input");
    if (audio.sample_rate != options.expected_sample_rate || audio.samples.empty()) throw std::runtime_error("worker WAV has unexpected sample rate or no samples");
    const double source_audio_seconds = options.source_duration_seconds - options.stream_start_seconds;
    if (!finite_positive(source_audio_seconds)) throw std::runtime_error("selected audio stream starts after the source timeline");
    const long source_samples = static_cast<long>(std::floor(
        source_audio_seconds * audio.sample_rate + 1e-6
    ));
    if (source_samples <= 0) throw std::runtime_error("selected audio stream has no source-timeline samples");
    if (static_cast<long>(audio.samples.size()) > source_samples) {
        audio.samples.resize(static_cast<size_t>(source_samples));
    }
    if (audio.samples.empty()) throw std::runtime_error("selected audio stream has no samples after timeline clamp");
    const long window_samples = exact_sample_count(
        options.window_seconds, audio.sample_rate, "--window-seconds"
    );
    const long hop_samples = exact_sample_count(
        options.hop_seconds, audio.sample_rate, "--hop-seconds"
    );
    const long planned_windows = static_cast<long>(std::ceil(
        static_cast<double>(audio.samples.size()) / hop_samples
    ));
    if (planned_windows > options.max_windows) {
        throw std::runtime_error("source exceeds configured CED window cap");
    }
    std::vector<std::pair<double, std::vector<float>>> outputs;
    std::vector<double> ends, padded;
    for (long window = 0, start = 0; start < static_cast<long>(audio.samples.size()); ++window, start += hop_samples) {
        if (!parent_alive(options.parent_pid)) return kParentLost;
        if (window >= options.max_windows) throw std::runtime_error("source exceeds configured CED window cap");
        const long end = std::min(start + window_samples, static_cast<long>(audio.samples.size()));
        std::vector<float> sample(audio.samples.begin() + start, audio.samples.begin() + end);
        const double pad_seconds = std::max(0L, window_samples - static_cast<long>(sample.size())) / static_cast<double>(audio.sample_rate);
        sample.resize(static_cast<size_t>(window_samples), 0.0f);
        std::vector<float> logits, probabilities;
        if (!model.classify(sample, logits, probabilities)) throw std::runtime_error("CED inference failed");
        if (probabilities.size() != model.config().labels.size()) throw std::runtime_error("CED label and score count differ");
        outputs.emplace_back(options.stream_start_seconds + static_cast<double>(start) / audio.sample_rate, std::move(probabilities));
        ends.push_back(options.stream_start_seconds + static_cast<double>(end) / audio.sample_rate);
        padded.push_back(pad_seconds);
    }
    if (outputs.empty()) throw std::runtime_error("CED did not produce source windows");
    if (!write_result(options, model, audio, model_sha, outputs, ends, padded)) {
        return kParentLost;
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && !std::strcmp(argv[1], "--version") && !std::strcmp(argv[2], "--json")) { version(); return 0; }
        if (argc > 2 && !std::strcmp(argv[1], "--probe")) return probe(argc, argv);
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
