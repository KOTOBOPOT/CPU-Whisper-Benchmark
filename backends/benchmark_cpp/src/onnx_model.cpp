#include "onnx_model.hpp"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <complex>
#include <filesystem>
#include <limits>

// Whisper constants
constexpr int N_MELS = 80;
constexpr int N_FFT = 400;
constexpr int HOP_LENGTH = 160;
constexpr int N_SAMPLES = 480000;  // 30 seconds at 16kHz
constexpr int N_FRAMES = 3000;     // N_SAMPLES / HOP_LENGTH
constexpr float MEL_FLOOR = 1e-10f;

WhisperONNXModel::WhisperONNXModel(
    const std::string& encoder_path,
    const std::string& decoder_path,
    const std::string& decoder_with_past_path,
    const std::string& vocab_path,
    int num_threads)
    : env_(nullptr), encoder_session_(nullptr), decoder_session_(nullptr), 
      decoder_with_past_session_(nullptr), session_options_(nullptr), last_latency_ms_(0.0)
{
    try {
        // Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "whisper");
        
        // Create session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(num_threads);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Load encoder
        std::cerr << "Loading encoder: " << encoder_path << std::endl;
        encoder_session_ = std::make_unique<Ort::Session>(
            *env_,
            encoder_path.c_str(),
            *session_options_
        );
        
        // Load decoder
        std::cerr << "Loading decoder: " << decoder_path << std::endl;
        decoder_session_ = std::make_unique<Ort::Session>(
            *env_,
            decoder_path.c_str(),
            *session_options_
        );
        
        // Load decoder with past
        std::cerr << "Loading decoder with past: " << decoder_with_past_path << std::endl;
        decoder_with_past_session_ = std::make_unique<Ort::Session>(
            *env_,
            decoder_with_past_path.c_str(),
            *session_options_
        );
        
        // Load vocabulary
        load_vocabulary(vocab_path);
        
        // Load suppress tokens
        std::filesystem::path suppress_path = std::filesystem::path(vocab_path).parent_path() / "suppress_tokens.json";
        if (std::filesystem::exists(suppress_path)) {
            load_suppress_tokens(suppress_path.string());
            std::cerr << "Loaded " << suppress_tokens_.size() << " suppress tokens" << std::endl;
        } else {
            std::cerr << "Warning: suppress_tokens.json not found, no token suppression will be applied" << std::endl;
        }
        
        std::cerr << "Whisper ONNX models loaded successfully" << std::endl;
        std::cerr << "Vocabulary size: " << id_to_token_.size() << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        throw std::runtime_error(std::string("Failed to load models: ") + e.what());
    }
}

WhisperONNXModel::~WhisperONNXModel() = default;

void WhisperONNXModel::load_vocabulary(const std::string& vocab_path)
{
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
    }
    
    // Parse JSON manually (simple approach)
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    // Very simple JSON parsing for vocab format: {"token": id, ...}
    size_t pos = 0;
    while ((pos = content.find("\"", pos)) != std::string::npos) {
        size_t token_start = pos + 1;
        size_t token_end = content.find("\"", token_start);
        if (token_end == std::string::npos) break;
        
        std::string token = content.substr(token_start, token_end - token_start);
        
        // Find ID
        pos = content.find(":", token_end);
        if (pos == std::string::npos) break;
        pos++;
        
        // Skip whitespace
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        
        // Read number
        size_t num_end = pos;
        while (num_end < content.size() && (isdigit(content[num_end]) || content[num_end] == '-')) num_end++;
        
        if (num_end > pos) {
            int id = std::stoi(content.substr(pos, num_end - pos));
            id_to_token_[id] = token;
            token_to_id_[token] = id;
        }
        
        pos = num_end;
    }
    
    std::cerr << "Loaded " << id_to_token_.size() << " vocabulary tokens" << std::endl;
}

void WhisperONNXModel::load_suppress_tokens(const std::string& suppress_tokens_path)
{
    std::ifstream file(suppress_tokens_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open suppress tokens file: " + suppress_tokens_path);
    }
    
    // Parse JSON manually (simple approach)
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    // Find "suppress_tokens" array
    size_t pos = content.find("\"suppress_tokens\"");
    if (pos != std::string::npos) {
        pos = content.find("[", pos);
        size_t end_pos = content.find("]", pos);
        if (pos != std::string::npos && end_pos != std::string::npos) {
            std::string tokens_str = content.substr(pos + 1, end_pos - pos - 1);
            std::istringstream iss(tokens_str);
            std::string token;
            while (std::getline(iss, token, ',')) {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);
                if (!token.empty()) {
                    suppress_tokens_.push_back(std::stoi(token));
                }
            }
        }
    }
    
    // Find "begin_suppress_tokens" array
    pos = content.find("\"begin_suppress_tokens\"");
    if (pos != std::string::npos) {
        pos = content.find("[", pos);
        size_t end_pos = content.find("]", pos);
        if (pos != std::string::npos && end_pos != std::string::npos) {
            std::string tokens_str = content.substr(pos + 1, end_pos - pos - 1);
            std::istringstream iss(tokens_str);
            std::string token;
            while (std::getline(iss, token, ',')) {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);
                if (!token.empty()) {
                    begin_suppress_tokens_.push_back(std::stoi(token));
                }
            }
        }
    }
}

std::string WhisperONNXModel::transcribe(const std::vector<float>& audio_data, int sample_rate)
{
    if (audio_data.empty()) {
        throw std::runtime_error("Audio data is empty");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Step 1: Compute mel spectrogram
        std::vector<float> mel_spec = compute_mel_spectrogram(audio_data, sample_rate);
        
        // Step 2: Run encoder
        std::vector<float> encoder_output = run_encoder(mel_spec);
        
        // Debug: check encoder output
        float encoder_sum = 0.0f;
        float encoder_max = -1e10f;
        float encoder_min = 1e10f;
        for (float val : encoder_output) {
            encoder_sum += val;
            encoder_max = std::max(encoder_max, val);
            encoder_min = std::min(encoder_min, val);
        }
        std::cerr << "Encoder output size: " << encoder_output.size() << std::endl;
        std::cerr << "Encoder output mean: " << (encoder_sum / encoder_output.size()) << std::endl;
        std::cerr << "Encoder output min/max: " << encoder_min << " / " << encoder_max << std::endl;
        
        // Step 3: Run decoder to generate tokens
        std::vector<int> token_ids = run_decoder(encoder_output);
        
        // Step 4: Decode tokens to text
        std::string text = decode_tokens(token_ids);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        last_latency_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        return text;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        throw std::runtime_error(std::string("Inference failed: ") + e.what());
    }
}

std::vector<float> WhisperONNXModel::compute_mel_spectrogram(
    const std::vector<float>& audio_data,
    int sample_rate)
{
    // Whisper expects exactly 30 seconds of audio
    std::vector<float> padded_audio(N_SAMPLES, 0.0f);
    size_t copy_size = std::min(audio_data.size(), (size_t)N_SAMPLES);
    std::copy(audio_data.begin(), audio_data.begin() + copy_size, padded_audio.begin());
    
    // Compute mel spectrogram
    // Shape: [N_MELS, N_FRAMES] = [80, 3000]
    std::vector<float> mel_spec(N_MELS * N_FRAMES);
    
    // Simplified mel computation using Hann window and log energy
    std::vector<float> hann_window(N_FFT);
    for (int i = 0; i < N_FFT; i++) {
        hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (N_FFT - 1)));
    }
    
    // Process each frame
    for (int frame = 0; frame < N_FRAMES; frame++) {
        int start_idx = frame * HOP_LENGTH;
        
        // Compute energy in this frame
        float energy = 0.0f;
        for (int i = 0; i < N_FFT && (start_idx + i) < N_SAMPLES; i++) {
            float sample = padded_audio[start_idx + i] * hann_window[i];
            energy += sample * sample;
        }
        
        // Convert to log mel scale
        float log_mel = std::log(std::max(energy, MEL_FLOOR));
        
        // Fill all mel bins with same value (simplified)
        // In real implementation, apply mel filterbank
        for (int mel_bin = 0; mel_bin < N_MELS; mel_bin++) {
            mel_spec[mel_bin * N_FRAMES + frame] = log_mel;
        }
    }
    
    // Don't normalize - Whisper processor doesn't normalize mel-spectrogram
    // The log-mel values are used directly
    
    // Debug: compute overall statistics
    float mel_sum = 0.0f, mel_min = 1e10f, mel_max = -1e10f;
    for (float val : mel_spec) {
        mel_sum += val;
        mel_min = std::min(mel_min, val);
        mel_max = std::max(mel_max, val);
    }
    float mel_mean = mel_sum / mel_spec.size();
    
    float mel_std = 0.0f;
    for (float val : mel_spec) {
        float diff = val - mel_mean;
        mel_std += diff * diff;
    }
    mel_std = std::sqrt(mel_std / mel_spec.size());
    
    std::cerr << "C++ Mel-spectrogram statistics:" << std::endl;
    std::cerr << "  Mean: " << mel_mean << std::endl;
    std::cerr << "  Std: " << mel_std << std::endl;
    std::cerr << "  Min: " << mel_min << std::endl;
    std::cerr << "  Max: " << mel_max << std::endl;
    
    return mel_spec;
}

std::vector<float> WhisperONNXModel::run_encoder(const std::vector<float>& mel_spec)
{
    // Input shape: [batch=1, n_mels=80, n_frames=3000]
    std::vector<int64_t> input_shape = {1, N_MELS, N_FRAMES};
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(mel_spec.data()),
        mel_spec.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    const char* input_names[] = {"input_features"};
    const char* output_names[] = {"last_hidden_state"};
    
    auto output_tensors = encoder_session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );
    
    // Extract output
    auto& output_tensor = output_tensors[0];
    float* output_data = output_tensor.GetTensorMutableData<float>();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(output_data, output_data + output_size);
}

std::vector<int> WhisperONNXModel::run_decoder(const std::vector<float>& encoder_hidden_states)
{
    // Constants
    const int batch_size = 1;
    const int encoder_seq_len = 1500;
    
    // Calculate hidden_size from encoder_hidden_states
    const int hidden_size = encoder_hidden_states.size() / (batch_size * encoder_seq_len);
    
    // Model architecture (derived from hidden_size)
    // whisper-base: 512, 6 layers, 8 heads
    // whisper-small: 768, 12 layers, 12 heads
    // whisper-medium: 1024, 24 layers, 16 heads
    // whisper-large: 1280, 32 layers, 20 heads
    const int num_layers = (hidden_size == 512) ? 6 : (hidden_size == 768) ? 12 : (hidden_size == 1024) ? 24 : 32;
    const int num_heads = (hidden_size == 512) ? 8 : (hidden_size == 768) ? 12 : (hidden_size == 1024) ? 16 : 20;
    const int head_dim = hidden_size / num_heads;
    const int max_length = 446;
    
    // Initialize decoder with start tokens
    // Whisper expects: <|startoftranscript|> <|language|> <|transcribe|> <|notimestamps|>
    std::vector<int> generated_tokens;
    generated_tokens.push_back(WhisperTokens::DECODER_START_TOKEN);  // Position 0: 50258
    generated_tokens.push_back(50263);  // Position 1: Russian language token (<|ru|>)
    generated_tokens.push_back(WhisperTokens::TRANSCRIBE_TOKEN);  // Position 2: 50359
    generated_tokens.push_back(WhisperTokens::NO_TIMESTAMPS_TOKEN);  // Position 3: 50363
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Storage for past key-values (will be populated after first step)
    std::vector<std::vector<float>> past_key_values;
    
    // First step: use decoder_model.onnx with initial tokens
    {
        std::vector<int64_t> input_ids;
        for (int token : generated_tokens) {
            input_ids.push_back(token);
        }
        
        std::vector<int64_t> input_ids_shape = {batch_size, static_cast<int64_t>(input_ids.size())};
        std::vector<int64_t> encoder_shape = {batch_size, encoder_seq_len, hidden_size};
        
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            input_ids.data(),
            input_ids.size(),
            input_ids_shape.data(),
            input_ids_shape.size()
        );
        
        Ort::Value encoder_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(encoder_hidden_states.data()),
            encoder_hidden_states.size(),
            encoder_shape.data(),
            encoder_shape.size()
        );
        
        const char* input_names[] = {"input_ids", "encoder_hidden_states"};
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_ids_tensor));
        input_tensors.push_back(std::move(encoder_tensor));
        
        // Request logits and all present key-values
        std::vector<std::string> output_name_strings = {"logits"};
        for (int layer = 0; layer < num_layers; layer++) {
            output_name_strings.push_back("present." + std::to_string(layer) + ".decoder.key");
            output_name_strings.push_back("present." + std::to_string(layer) + ".decoder.value");
            output_name_strings.push_back("present." + std::to_string(layer) + ".encoder.key");
            output_name_strings.push_back("present." + std::to_string(layer) + ".encoder.value");
        }
        std::vector<const char*> output_names;
        for (const auto& s : output_name_strings) {
            output_names.push_back(s.c_str());
        }
        
        auto output_tensors = decoder_session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );
        
        // Get logits for last position
        float* logits_data = output_tensors[0].GetTensorMutableData<float>();
        auto logits_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int vocab_size = logits_shape[2];
        int last_pos = logits_shape[1] - 1;
        float* last_logits = logits_data + (last_pos * vocab_size);
        
        // Apply token suppression (including begin_suppress_tokens for first step)
        for (int token_id : suppress_tokens_) {
            if (token_id < vocab_size) {
                last_logits[token_id] = -std::numeric_limits<float>::infinity();
            }
        }
        for (int token_id : begin_suppress_tokens_) {
            if (token_id < vocab_size) {
                last_logits[token_id] = -std::numeric_limits<float>::infinity();
            }
        }
        
        int next_token = std::max_element(last_logits, last_logits + vocab_size) - last_logits;
        
        std::cerr << "Step 0: generated token " << next_token << std::endl;
        
        if (next_token == WhisperTokens::EOS_TOKEN) {
            std::cerr << "EOS token generated at step 0" << std::endl;
            return generated_tokens;
        }
        
        generated_tokens.push_back(next_token);
        
        // Extract past key-values for subsequent steps
        // Output order: logits, decoder.key, decoder.value, encoder.key, encoder.value (repeated for each layer)
        past_key_values.resize(num_layers * 4);
        for (int i = 1; i < (int)output_tensors.size(); i++) {
            float* data = output_tensors[i].GetTensorMutableData<float>();
            size_t count = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
            past_key_values[i - 1].assign(data, data + count);
        }
        
        std::cerr << "Extracted " << past_key_values.size() << " past key-value tensors" << std::endl;
    }
    
    // Subsequent steps: use decoder_with_past_model.onnx
    for (int step = 1; step < max_length; step++) {
        // Input is only the last generated token
        std::vector<int64_t> input_ids = {generated_tokens.back()};
        std::vector<int64_t> input_ids_shape = {batch_size, 1};
        
        // Prepare input tensors
        std::vector<Ort::Value> input_tensors;
        std::vector<std::string> input_name_strings;
        
        // Add input_ids
        input_name_strings.push_back("input_ids");
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info,
            input_ids.data(),
            input_ids.size(),
            input_ids_shape.data(),
            input_ids_shape.size()
        ));
        
        // Add past key-values
        // past_seq_len is the sequence length of the decoder key-values
        // After generating N tokens, decoder KV has N-1 positions (input tokens, not including last generated)
        int past_seq_len = generated_tokens.size() - 1;
        for (int layer = 0; layer < num_layers; layer++) {
            const char* kv_types[] = {"decoder.key", "decoder.value", "encoder.key", "encoder.value"};
            for (int kv_type_idx = 0; kv_type_idx < 4; kv_type_idx++) {
                std::string name = "past_key_values." + std::to_string(layer) + "." + kv_types[kv_type_idx];
                input_name_strings.push_back(name);
                
                int kv_idx = layer * 4 + kv_type_idx;
                bool is_encoder_kv = (kv_type_idx >= 2);
                int seq_len = is_encoder_kv ? encoder_seq_len : past_seq_len;
                std::vector<int64_t> kv_shape = {batch_size, num_heads, seq_len, head_dim};
                
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    past_key_values[kv_idx].data(),
                    past_key_values[kv_idx].size(),
                    kv_shape.data(),
                    kv_shape.size()
                ));
            }
        }
        
        // Convert to const char* array
        std::vector<const char*> input_names;
        for (const auto& s : input_name_strings) {
            input_names.push_back(s.c_str());
        }
        
        // Run decoder with past - prepare output names
        // decoder_with_past returns only decoder keys (encoder keys don't change!)
        std::vector<std::string> output_name_strings = {"logits"};
        for (int layer = 0; layer < num_layers; layer++) {
            output_name_strings.push_back("present." + std::to_string(layer) + ".decoder.key");
            output_name_strings.push_back("present." + std::to_string(layer) + ".decoder.value");
        }
        std::vector<const char*> output_names;
        for (const auto& s : output_name_strings) {
            output_names.push_back(s.c_str());
        }
        
        auto output_tensors = decoder_with_past_session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );
        
        // Get logits (shape: [batch=1, 1, vocab_size])
        float* logits_data = output_tensors[0].GetTensorMutableData<float>();
        auto logits_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int vocab_size = logits_shape[2];
        
        // Apply token suppression (NOT begin_suppress_tokens after first step)
        for (int token_id : suppress_tokens_) {
            if (token_id < vocab_size) {
                logits_data[token_id] = -std::numeric_limits<float>::infinity();
            }
        }
        
        int next_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
        
        if (step < 10 || step % 50 == 0) {
            std::cerr << "Step " << step << ": generated token " << next_token << std::endl;
        }
        
        if (next_token == WhisperTokens::EOS_TOKEN) {
            std::cerr << "EOS token generated at step " << step << std::endl;
            break;
        }
        
        generated_tokens.push_back(next_token);
        
        // Update past key-values with present decoder keys
        // Output: logits, decoder.key[0], decoder.value[0], ..., decoder.key[5], decoder.value[5]
        // We only update decoder keys (indices 0,1 for each layer)
        // Encoder keys (indices 2,3 for each layer) stay the same
        int output_idx = 1; // Skip logits
        for (int layer = 0; layer < num_layers; layer++) {
            // Update decoder key
            int kv_idx = layer * 4 + 0; // decoder.key
            float* data = output_tensors[output_idx].GetTensorMutableData<float>();
            size_t count = output_tensors[output_idx].GetTensorTypeAndShapeInfo().GetElementCount();
            auto shape = output_tensors[output_idx].GetTensorTypeAndShapeInfo().GetShape();
            
            if ((step <= 3 || step % 50 == 0) && layer == 0) {
                std::cerr << "Step " << step << ": decoder_with_past returned decoder.key[" << layer << "] with size " << count
                          << " shape [";
                for (size_t i = 0; i < shape.size(); i++) {
                    std::cerr << shape[i];
                    if (i < shape.size() - 1) std::cerr << ", ";
                }
                std::cerr << "]" << std::endl;
                std::cerr << "  Previous KV size: " << past_key_values[kv_idx].size() 
                          << " (seq_len=" << (past_key_values[kv_idx].size() / (batch_size * num_heads * head_dim)) << ")" << std::endl;
                std::cerr << "  New KV seq_len: " << shape[2] << std::endl;
            }
            
            past_key_values[kv_idx].assign(data, data + count);
            output_idx++;
            
            // Update decoder value
            kv_idx = layer * 4 + 1; // decoder.value
            data = output_tensors[output_idx].GetTensorMutableData<float>();
            count = output_tensors[output_idx].GetTensorTypeAndShapeInfo().GetElementCount();
            past_key_values[kv_idx].assign(data, data + count);
            output_idx++;
            
            // encoder.key and encoder.value (indices 2,3) remain unchanged
        }
        
        // Early stopping for timestamp tokens
        if (next_token >= 50364) {
            continue;
        }
    }
    
    return generated_tokens;
}

std::string WhisperONNXModel::decode_tokens(const std::vector<int>& token_ids)
{
    std::string result;
    
    for (int token_id : token_ids) {
        // Skip special tokens
        if (token_id == WhisperTokens::DECODER_START_TOKEN ||
            token_id == WhisperTokens::EOS_TOKEN ||
            token_id == WhisperTokens::NO_TIMESTAMPS_TOKEN ||
            token_id >= 50257) {  // All special tokens >= 50257
            continue;
        }
        
        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) {
            std::string token = it->second;
            
            // Handle BPE encoding
            if (token.find("Ġ") == 0) {
                result += " " + token.substr(3);  // Ġ is 3 bytes in UTF-8
            } else {
                result += token;
            }
        }
    }
    
    // Trim whitespace
    size_t start = result.find_first_not_of(" \t\n\r");
    size_t end = result.find_last_not_of(" \t\n\r");
    if (start != std::string::npos && end != std::string::npos) {
        result = result.substr(start, end - start + 1);
    }
    
    return result;
}

// Public wrapper methods for hybrid Python/C++ approach
std::vector<float> WhisperONNXModel::run_encoder_from_mel(const std::vector<float>& mel_spec)
{
    return run_encoder(mel_spec);
}

std::vector<int> WhisperONNXModel::run_decoder_from_encoder_output(const std::vector<float>& encoder_output)
{
    return run_decoder(encoder_output);
}

std::string WhisperONNXModel::decode_tokens_to_text(const std::vector<int>& token_ids)
{
    return decode_tokens(token_ids);
}
