#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>

namespace Ort {
    class Session;
    class Env;
    class SessionOptions;
    class Value;
}

// Whisper special token IDs
struct WhisperTokens {
    static constexpr int DECODER_START_TOKEN = 50258;
    static constexpr int EOS_TOKEN = 50257;
    static constexpr int NO_TIMESTAMPS_TOKEN = 50363;
    static constexpr int TRANSCRIBE_TOKEN = 50359;  // <|transcribe|>
    static constexpr int TRANSLATE_TOKEN = 50358;   // <|translate|>
    static constexpr int EN_TOKEN = 50259;          // <|en|>
};

/**
 * Full Whisper model with encoder and decoder
 */
class WhisperONNXModel {
public:
    /**
     * Constructor
     * @param encoder_path Path to encoder ONNX model
     * @param decoder_path Path to decoder ONNX model
     * @param decoder_with_past_path Path to decoder with past ONNX model
     * @param vocab_path Path to vocab.json file
     * @param num_threads Number of threads for inference
     */
    explicit WhisperONNXModel(
        const std::string& encoder_path,
        const std::string& decoder_path,
        const std::string& decoder_with_past_path,
        const std::string& vocab_path,
        int num_threads = 4
    );
    
    ~WhisperONNXModel();
    
    /**
     * Transcribe audio
     * @param audio_data Audio samples (16kHz, mono)
     * @param sample_rate Sample rate (should be 16000)
     * @return Transcribed text
     */
    std::string transcribe(const std::vector<float>& audio_data, int sample_rate = 16000);
    
    /**
     * Get last inference time in milliseconds
     */
    double get_last_latency_ms() const { return last_latency_ms_; }
    
    /**
     * Check if models are loaded
     */
    bool is_loaded() const { 
        return encoder_session_ != nullptr && 
               decoder_session_ != nullptr && 
               decoder_with_past_session_ != nullptr; 
    }
    
    /**
     * Run encoder with pre-computed mel-spectrogram
     */
    std::vector<float> run_encoder_from_mel(const std::vector<float>& mel_spec);
    
    /**
     * Run decoder with encoder output
     */
    std::vector<int> run_decoder_from_encoder_output(const std::vector<float>& encoder_output);
    
    /**
     * Decode token IDs to text
     */
    std::string decode_tokens_to_text(const std::vector<int>& token_ids);
    
    /**
     * Compute mel-spectrogram from audio (for C++ mel-spectrogram path)
     */
    std::vector<float> compute_mel_spectrogram(const std::vector<float>& audio_data, int sample_rate);

private:
    // ONNX Runtime objects
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    std::unique_ptr<Ort::Session> decoder_with_past_session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    
    // Vocabulary for decoding
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    
    // Token suppression for generation
    std::vector<int> suppress_tokens_;
    std::vector<int> begin_suppress_tokens_;
    
    double last_latency_ms_ = 0.0;
    
    /**
     * Load vocabulary from JSON file
     */
    void load_vocabulary(const std::string& vocab_path);
    
    /**
     * Load suppress tokens from JSON file
     */
    void load_suppress_tokens(const std::string& suppress_tokens_path);
    
    /**
     * Run encoder: mel_spec -> hidden_states
     */
    std::vector<float> run_encoder(const std::vector<float>& mel_spec);
    
    /**
     * Run decoder loop: hidden_states -> token_ids
     */
    std::vector<int> run_decoder(const std::vector<float>& encoder_hidden_states);
    
    /**
     * Decode token IDs to text
     */
    std::string decode_tokens(const std::vector<int>& token_ids);
};
