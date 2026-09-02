#include "vllm_client.h"

#include <stdexcept>

// Header-only LLM client from VideoPipe third_party.
// It wraps cpp_httplib and needs OpenSSL at link time. VideoPipe adds the
// OpenSSL link deps to its own target when VP_WITH_LLM is ON.
#include <llmlib.hpp>

namespace trashbot {

std::string analyse(const VllmParams& params, const nlohmann::json& messages) {
    if (params.endpoint_url.rfind("http://", 0) != 0
            && params.endpoint_url.rfind("https://", 0) != 0) {
        throw std::runtime_error(
            "vLLM endpoint_url must start with http:// or https://: "
            + params.endpoint_url);
    }
    if (params.model.empty()) {
        throw std::runtime_error("vLLM model name is empty");
    }
    if (!messages.is_array()) {
        throw std::runtime_error("vLLM messages must be a JSON array");
    }

    // Sampling options merge into the top level of the request payload.
    nlohmann::json options = {
        {"temperature", params.temperature},
        {"max_tokens", params.max_tokens},
        {"seed", params.seed},
    };

    llmlib::LLMClient client(params.endpoint_url, params.api_key,
                             llmlib::LLMBackendType::OpenAI);
    client.set_connection_timeout(params.timeout_s);

    std::string content;
    try {
        content = client.chat(params.model, messages, options);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("vLLM chat request threw: ")
                                 + e.what());
    }

    // llmlib maps transport errors and non-200 answers to an empty result.
    // We turn that into a clear exception here.
    if (content.empty()) {
        throw std::runtime_error(
            "vLLM chat returned no content. Check the server status, the "
            "model name, and the api key. Endpoint: " + params.endpoint_url
            + ", model: " + params.model);
    }
    return content;
}

}  // namespace trashbot
