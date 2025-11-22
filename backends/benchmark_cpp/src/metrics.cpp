#include "metrics.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <cmath>

int MetricsCalculator::edit_distance(
    const std::vector<std::string>& reference,
    const std::vector<std::string>& hypothesis)
{
    size_t m = reference.size();
    size_t n = hypothesis.size();
    
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    
    for (size_t i = 0; i <= m; i++) {
        dp[i][0] = i;
    }
    for (size_t j = 0; j <= n; j++) {
        dp[0][j] = j;
    }
    
    for (size_t i = 1; i <= m; i++) {
        for (size_t j = 1; j <= n; j++) {
            int substitution_cost = (reference[i - 1] == hypothesis[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({
                dp[i - 1][j] + 1,          // deletion
                dp[i][j - 1] + 1,          // insertion
                dp[i - 1][j - 1] + substitution_cost  // substitution
            });
        }
    }
    
    return dp[m][n];
}

std::string MetricsCalculator::normalize_text(const std::string& text)
{
    std::string normalized;
    bool in_space = true;
    
    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isspace(uc)) {
            if (!in_space) {
                normalized += ' ';
                in_space = true;
            }
        } else {
            normalized += std::tolower(uc);
            in_space = false;
        }
    }
    
    // Trim trailing space
    if (!normalized.empty() && normalized.back() == ' ') {
        normalized.pop_back();
    }
    
    return normalized;
}

std::vector<std::string> MetricsCalculator::split_words(const std::string& text)
{
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        if (!word.empty()) {
            words.push_back(word);
        }
    }
    
    return words;
}

std::vector<std::string> MetricsCalculator::split_chars(const std::string& text)
{
    std::vector<std::string> chars;
    for (char c : text) {
        if (!std::isspace(c)) {
            chars.push_back(std::string(1, c));
        }
    }
    return chars;
}

std::pair<int, int> MetricsCalculator::calculate_wer(
    const std::string& reference,
    const std::string& hypothesis)
{
    std::string norm_ref = normalize_text(reference);
    std::string norm_hyp = normalize_text(hypothesis);
    
    std::vector<std::string> ref_words = split_words(norm_ref);
    std::vector<std::string> hyp_words = split_words(norm_hyp);
    
    int distance = edit_distance(ref_words, hyp_words);
    int ref_word_count = ref_words.size();
    
    return {distance, ref_word_count};
}

std::pair<int, int> MetricsCalculator::calculate_cer(
    const std::string& reference,
    const std::string& hypothesis)
{
    std::string norm_ref = normalize_text(reference);
    std::string norm_hyp = normalize_text(hypothesis);
    
    std::vector<std::string> ref_chars = split_chars(norm_ref);
    std::vector<std::string> hyp_chars = split_chars(norm_hyp);
    
    int distance = edit_distance(ref_chars, hyp_chars);
    int ref_char_count = ref_chars.size();
    
    return {distance, ref_char_count};
}

void BenchmarkStats::update_success(
    double latency_ms,
    int wer_distance,
    int ref_words,
    int cer_distance,
    int ref_chars)
{
    total_latency_ms += latency_ms;
    processed++;
    wer_distance += wer_distance;
    wer_ref_words += ref_words;
    cer_distance += cer_distance;
    cer_ref_chars += ref_chars;
}

void BenchmarkStats::update_failure()
{
    failed++;
}

double BenchmarkStats::average_latency() const
{
    if (processed == 0) return 0.0;
    return total_latency_ms / processed;
}

double BenchmarkStats::wer() const
{
    if (wer_ref_words == 0) return 0.0;
    return static_cast<double>(wer_distance) / static_cast<double>(wer_ref_words);
}

double BenchmarkStats::cer() const
{
    if (cer_ref_chars == 0) return 0.0;
    return static_cast<double>(cer_distance) / static_cast<double>(cer_ref_chars);
}
