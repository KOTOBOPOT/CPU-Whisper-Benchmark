#pragma once

#include <string>
#include <vector>
#include <cstdint>

/**
 * Structure for single transcription result
 */
struct TranscriptionResult {
    std::string filename;
    std::string reference;
    std::string hypothesis;
    double latency_ms;
    std::string status;
    std::string error;
};

/**
 * Metrics calculator for WER, CER, and statistics
 */
class MetricsCalculator {
public:
    /**
     * Compute edit distance (Levenshtein distance) between two sequences
     */
    static int edit_distance(
        const std::vector<std::string>& reference,
        const std::vector<std::string>& hypothesis
    );
    
    /**
     * Normalize text for comparison
     */
    static std::string normalize_text(const std::string& text);
    
    /**
     * Split text into words
     */
    static std::vector<std::string> split_words(const std::string& text);
    
    /**
     * Split text into characters
     */
    static std::vector<std::string> split_chars(const std::string& text);
    
    /**
     * Calculate WER (Word Error Rate)
     * @return {wer_distance, ref_words}
     */
    static std::pair<int, int> calculate_wer(
        const std::string& reference,
        const std::string& hypothesis
    );
    
    /**
     * Calculate CER (Character Error Rate)
     * @return {cer_distance, ref_chars}
     */
    static std::pair<int, int> calculate_cer(
        const std::string& reference,
        const std::string& hypothesis
    );
};

/**
 * Running statistics for benchmark results
 */
class BenchmarkStats {
public:
    void update_success(
        double latency_ms,
        int wer_distance,
        int ref_words,
        int cer_distance,
        int ref_chars
    );
    
    void update_failure();
    
    double average_latency() const;
    double wer() const;
    double cer() const;
    
    double total_latency_ms = 0.0;
    int processed = 0;
    int failed = 0;
    int wer_distance = 0;
    int wer_ref_words = 0;
    int cer_distance = 0;
    int cer_ref_chars = 0;
};
