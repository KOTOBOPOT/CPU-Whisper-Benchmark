#pragma once

#include <vector>
#include <string>
#include <cstdint>

/**
 * Audio utilities for loading and processing audio files
 */
class AudioUtils {
public:
    /**
     * Load audio from file (WAV, MP3, OGG, OPUS, etc.)
     * @param file_path Path to audio file
     * @param target_sample_rate Target sample rate (resampling if needed)
     * @return Audio samples as float array (normalized to [-1, 1])
     */
    static std::vector<float> load_audio_file(
        const std::string& file_path,
        int target_sample_rate = 16000
    );
    
    /**
     * Load audio from raw WAV file
     * @param file_path Path to WAV file
     * @return Audio samples as float array
     */
    static std::vector<float> load_wav_file(const std::string& file_path);
    
    /**
     * Load audio from OPUS file
     * @param file_path Path to OPUS file
     * @return Audio samples as float array
     */
    static std::vector<float> load_opus_file(const std::string& file_path);
    
    /**
     * Normalize audio samples to [-1, 1] range
     */
    static void normalize_audio(std::vector<float>& audio);
    
    /**
     * Resample audio to target sample rate
     * @param audio Input audio samples
     * @param input_sr Input sample rate
     * @param output_sr Target sample rate
     * @return Resampled audio
     */
    static std::vector<float> resample_audio(
        const std::vector<float>& audio,
        int input_sr,
        int output_sr
    );
    
    /**
     * Get audio duration in seconds
     */
    static double get_audio_duration(
        const std::vector<float>& audio,
        int sample_rate
    );
    
    /**
     * Check if file exists
     */
    static bool file_exists(const std::string& file_path);
};
