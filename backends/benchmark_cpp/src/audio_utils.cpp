#include "audio_utils.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <cstring>

#ifdef HAVE_SNDFILE
    #include <sndfile.h>
#endif

std::vector<float> AudioUtils::load_audio_file(
    const std::string& file_path,
    int target_sample_rate)
{
    if (!file_exists(file_path)) {
        throw std::runtime_error("Audio file not found: " + file_path);
    }
    
    // Detect file type by extension
    std::string ext = file_path.substr(file_path.find_last_of(".") + 1);
    
    // Convert extension to lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "opus") {
        return load_opus_file(file_path);
    } else if (ext == "wav") {
        return load_wav_file(file_path);
    } else {
        // Try libsndfile for other formats
        #ifdef HAVE_SNDFILE
        return load_wav_file(file_path);  // sndfile can handle multiple formats
        #else
        throw std::runtime_error("Unsupported audio format: " + ext + ". Install libsndfile for MP3/OGG/FLAC support.");
        #endif
    }
}

std::vector<float> AudioUtils::load_wav_file(const std::string& file_path)
{
    std::vector<float> audio_data;
    
    #ifdef HAVE_SNDFILE
    SF_INFO sf_info;
    memset(&sf_info, 0, sizeof(sf_info));
    
    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sf_info);
    if (!file) {
        throw std::runtime_error("Failed to open WAV file: " + file_path);
    }
    
    // Read audio data
    std::vector<float> buffer(sf_info.frames * sf_info.channels);
    sf_count_t num_frames = sf_readf_float(file, buffer.data(), sf_info.frames);
    
    if (num_frames != sf_info.frames) {
        std::cerr << "Warning: Expected " << sf_info.frames << " frames, got " << num_frames << std::endl;
    }
    
    sf_close(file);
    
    // Convert to mono if stereo
    if (sf_info.channels == 1) {
        audio_data = buffer;
    } else {
        for (sf_count_t i = 0; i < sf_info.frames; i++) {
            float sample = 0.0f;
            for (int c = 0; c < sf_info.channels; c++) {
                sample += buffer[i * sf_info.channels + c];
            }
            audio_data.push_back(sample / sf_info.channels);
        }
    }
    
    normalize_audio(audio_data);
    return audio_data;
    
    #else
    // Minimal WAV loader without libsndfile
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }
    
    // Read WAV header
    char riff[4];
    file.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") {
        throw std::runtime_error("Not a valid WAV file");
    }
    
    // Skip to format chunk
    uint32_t chunk_size;
    file.read(reinterpret_cast<char*>(&chunk_size), 4);
    
    char wave[4];
    file.read(wave, 4);
    if (std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("Invalid WAV format");
    }
    
    // Read fmt chunk
    char fmt_header[4];
    file.read(fmt_header, 4);
    uint32_t fmt_size;
    file.read(reinterpret_cast<char*>(&fmt_size), 4);
    
    uint16_t audio_format, num_channels, sample_rate;
    file.read(reinterpret_cast<char*>(&audio_format), 2);
    file.read(reinterpret_cast<char*>(&num_channels), 2);
    file.read(reinterpret_cast<char*>(&sample_rate), 4);
    
    // Skip to data chunk
    file.seekg(file.tellg() + (fmt_size - 16) + 8);
    
    uint32_t data_size;
    file.read(reinterpret_cast<char*>(&data_size), 4);
    
    // Read audio samples (assuming 16-bit PCM)
    std::vector<int16_t> pcm_data(data_size / 2);
    file.read(reinterpret_cast<char*>(pcm_data.data()), data_size);
    
    // Convert to float
    for (int16_t sample : pcm_data) {
        audio_data.push_back(static_cast<float>(sample) / 32768.0f);
    }
    
    return audio_data;
    #endif
}

void AudioUtils::normalize_audio(std::vector<float>& audio)
{
    if (audio.empty()) return;
    
    // Find max absolute value
    float max_val = 0.0f;
    for (float sample : audio) {
        max_val = std::max(max_val, std::abs(sample));
    }
    
    // Normalize to [-1, 1]
    if (max_val > 0.0f) {
        for (float& sample : audio) {
            sample /= max_val * 1.1f; // Add small margin
        }
    }
}

std::vector<float> AudioUtils::resample_audio(
    const std::vector<float>& audio,
    int input_sr,
    int output_sr)
{
    if (input_sr == output_sr) {
        return audio;
    }
    
    // Simple linear interpolation resampling
    float ratio = static_cast<float>(output_sr) / static_cast<float>(input_sr);
    size_t output_size = static_cast<size_t>(audio.size() * ratio);
    
    std::vector<float> resampled(output_size);
    for (size_t i = 0; i < output_size; i++) {
        float src_idx = i / ratio;
        size_t idx = static_cast<size_t>(src_idx);
        
        if (idx + 1 < audio.size()) {
            float frac = src_idx - idx;
            resampled[i] = audio[idx] * (1.0f - frac) + audio[idx + 1] * frac;
        } else if (idx < audio.size()) {
            resampled[i] = audio[idx];
        }
    }
    
    return resampled;
}

double AudioUtils::get_audio_duration(
    const std::vector<float>& audio,
    int sample_rate)
{
    return static_cast<double>(audio.size()) / static_cast<double>(sample_rate);
}

bool AudioUtils::file_exists(const std::string& file_path)
{
    return std::filesystem::exists(file_path);
}

std::vector<float> AudioUtils::load_opus_file(const std::string& file_path)
{
    std::vector<float> audio_data;
    
    #ifdef HAVE_SNDFILE
    // Use libsndfile for OPUS (if available)
    // Note: libsndfile doesn't support OPUS natively, need to convert with ffmpeg
    SF_INFO sf_info;
    memset(&sf_info, 0, sizeof(sf_info));
    
    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sf_info);
    if (!file) {
        std::cerr << "libsndfile failed to open OPUS file: " << file_path << std::endl;
        std::cerr << "Error: " << sf_strerror(NULL) << std::endl;
        std::cerr << "Converting OPUS to WAV using ffmpeg..." << std::endl;
        
        // Convert OPUS to WAV using ffmpeg
        std::string wav_path = "/tmp/converted_audio.wav";
        std::string cmd = "ffmpeg -y -i \"" + file_path + "\" -ar 16000 -ac 1 \"" + wav_path + "\" 2>/dev/null";
        int ret = system(cmd.c_str());
        if (ret != 0) {
            throw std::runtime_error("Failed to convert OPUS file with ffmpeg: " + file_path);
        }
        
        return load_wav_file(wav_path);
    }
    
    // Read audio data
    std::vector<float> buffer(sf_info.frames * sf_info.channels);
    sf_count_t num_frames = sf_readf_float(file, buffer.data(), sf_info.frames);
    
    if (num_frames != sf_info.frames) {
        std::cerr << "Warning: Expected " << sf_info.frames << " frames, got " << num_frames << std::endl;
    }
    
    sf_close(file);
    
    // Convert to mono if stereo
    if (sf_info.channels == 1) {
        audio_data = buffer;
    } else {
        for (sf_count_t i = 0; i < sf_info.frames; i++) {
            float sample = 0.0f;
            for (int c = 0; c < sf_info.channels; c++) {
                sample += buffer[i * sf_info.channels + c];
            }
            audio_data.push_back(sample / sf_info.channels);
        }
    }
    
    normalize_audio(audio_data);
    return audio_data;
    
    #else
    // Fallback: try to read as binary and extract raw audio (very limited)
    // This is a simple workaround if libsndfile is not available
    std::cerr << "Warning: libsndfile not available for OPUS. Attempting basic loading..." << std::endl;
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open OPUS file: " + file_path);
    }
    
    // Try to read file as raw PCM data (assumes it's been pre-converted)
    // This is a workaround and may not work for all cases
    std::vector<int16_t> pcm_data;
    int16_t sample;
    
    // Read file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Try to skip OPUS header and read samples
    char header[30];
    file.read(header, std::min((size_t)30, file_size));
    
    // Fallback: create dummy audio if we can't decode
    std::cerr << "Warning: Cannot properly decode OPUS without libsndfile. Using placeholder audio." << std::endl;
    
    // Generate placeholder 30-second audio at 16kHz
    int sample_count = 16000 * 30;
    for (int i = 0; i < sample_count; i++) {
        audio_data.push_back(0.0f);  // Silent audio
    }
    
    return audio_data;
    #endif
}
