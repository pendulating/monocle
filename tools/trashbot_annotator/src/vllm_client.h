#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace trashbot {

// Parameters for one call to an OpenAI-compatible server.
struct VllmParams {
    std::string endpoint_url;  // base url with scheme, example: http://localhost:8000/v1
    std::string model;
    std::string api_key;
    int max_tokens = 1024;
    double temperature = 0.0;
    long seed = 42;
    int timeout_s = 120;       // connect timeout in seconds
};

// Send one chat request. messages is a JSON array of message objects with
// role and content parts. Return the raw answer text from the first choice.
// Throw std::runtime_error on bad input or on a failed HTTP call.
std::string analyse(const VllmParams& params, const nlohmann::json& messages);

}  // namespace trashbot
