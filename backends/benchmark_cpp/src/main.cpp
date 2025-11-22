#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <thread>
#include <json/json.h>

#include "onnx_model.hpp"
#include "audio_utils.hpp"
#include "metrics.hpp"

namespace fs = std::filesystem;

struct ProgramArgs {
    std::string bench_name;
    std::string model_path;
    fs::path output_dir;
    fs::path data_root;
    int max_samples = -1;
    double sleep_between_requests = 0.0;
    int num_threads = 4;
    bool single_file_mode = false;
    std::string single_file_path;
    std::string output_format = "full";  // "full" or "json"
    bool interactive = false;  // Interactive mode for persistent process
};

class CSVWriter {
public:
    CSVWriter(const fs::path& path) : file_(path) {
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open CSV file: " + path.string());
        }
    }
    
    void write_header(const std::vector<std::string>& headers) {
        for (size_t i = 0; i < headers.size(); i++) {
            if (i > 0) file_ << ",";
            file_ << escape_csv(headers[i]);
        }
        file_ << "\n";
    }
    
    void write_row(const std::vector<std::string>& values) {
        for (size_t i = 0; i < values.size(); i++) {
            if (i > 0) file_ << ",";
            file_ << escape_csv(values[i]);
        }
        file_ << "\n";
    }
    
private:
    std::ofstream file_;
    
    std::string escape_csv(const std::string& field) {
        if (field.find(',') != std::string::npos ||
            field.find('"') != std::string::npos ||
            field.find('\n') != std::string::npos) {
            std::string escaped = "\"";
            for (char c : field) {
                if (c == '"') escaped += "\"\"";
                else escaped += c;
            }
            escaped += "\"";
            return escaped;
        }
        return field;
    }
};

std::vector<std::pair<std::string, std::string>> load_annotation(const fs::path& annotation_path)
{
    std::vector<std::pair<std::string, std::string>> items;
    
    std::ifstream file(annotation_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open annotation file: " + annotation_path.string());
    }
    
    std::string line;
    std::vector<std::string> headers;
    bool first_line = true;
    int filename_idx = -1;
    int text_idx = -1;
    
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            
            // Parse header
            std::istringstream iss(line);
            std::string header;
            int idx = 0;
            while (std::getline(iss, header, ',')) {
                if (header == "filename") filename_idx = idx;
                if (header == "text") text_idx = idx;
                headers.push_back(header);
                idx++;
            }
            
            if (filename_idx < 0 || text_idx < 0) {
                throw std::runtime_error("Annotation file must have 'filename' and 'text' columns");
            }
            continue;
        }
        
        // Parse data row
        std::vector<std::string> fields;
        std::istringstream iss(line);
        std::string field;
        while (std::getline(iss, field, ',')) {
            fields.push_back(field);
        }
        
        if (filename_idx < (int)fields.size() && text_idx < (int)fields.size()) {
            std::string filename = fields[filename_idx];
            std::string text = fields[text_idx];
            
            // Trim whitespace
            filename.erase(0, filename.find_first_not_of(" \t"));
            filename.erase(filename.find_last_not_of(" \t") + 1);
            text.erase(0, text.find_first_not_of(" \t"));
            text.erase(text.find_last_not_of(" \t") + 1);
            
            if (!filename.empty()) {
                items.push_back({filename, text});
            }
        }
    }
    
    return items;
}

ProgramArgs parse_args(int argc, char* argv[])
{
    ProgramArgs args;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--bench-name" && i + 1 < argc) {
            args.bench_name = argv[++i];
        } else if (arg == "--model-path" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--data-root" && i + 1 < argc) {
            args.data_root = argv[++i];
        } else if (arg == "--max-samples" && i + 1 < argc) {
            args.max_samples = std::stoi(argv[++i]);
        } else if (arg == "--sleep" && i + 1 < argc) {
            args.sleep_between_requests = std::stod(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            args.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--single-file" && i + 1 < argc) {
            args.single_file_mode = true;
            args.single_file_path = argv[++i];
        } else if (arg == "--output-format" && i + 1 < argc) {
            args.output_format = argv[++i];
        } else if (arg == "--interactive") {
            args.interactive = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --bench-name NAME        Benchmark name: golos_10_debug, golos, etc.\n"
                      << "  --model-path PATH        Path to ONNX model file (required)\n"
                      << "  --output-dir PATH        Output directory for results\n"
                      << "  --data-root PATH         Root directory with datasets (default: ./data)\n"
                      << "  --max-samples N          Maximum number of samples to process\n"
                      << "  --sleep SECONDS          Sleep between requests (seconds)\n"
                      << "  --threads N              Number of threads for model (default: 4)\n"
                      << "  --single-file PATH       Process a single audio file (service mode)\n"
                      << "  --interactive            Interactive mode for persistent process\n"
                      << "  --output-format FORMAT   Output format: full or json (default: full)\n"
                      << "  --help                   Show this help message\n\n"
                      << "Examples:\n"
                      << "  # Test with Russian speech data (OPUS format, 10 samples)\n"
                      << "  " << argv[0] << " --bench-name golos_10_debug --model-path model.onnx"
                      << " --output-dir ./results\n\n"
                      << "  # Test with subset of data\n"
                      << "  " << argv[0] << " --bench-name golos_10_debug --model-path model.onnx"
                      << " --output-dir ./results --max-samples 5\n\n"
                      << "Note: Supports WAV and OPUS audio formats. OPUS requires libsndfile.\n";
            std::exit(0);
        }
    }
    
    // Different requirements for different modes
    if (args.interactive || args.single_file_mode) {
        // Interactive and single-file modes only need model path
        if (args.model_path.empty()) {
            throw std::runtime_error("Missing required argument: --model-path");
        }
    } else {
        // Benchmark mode needs all arguments
        if (args.bench_name.empty() || args.model_path.empty() || args.output_dir.empty()) {
            throw std::runtime_error("Missing required arguments: --bench-name, --model-path, --output-dir");
        }
    }
    
    if (args.data_root.empty()) {
        args.data_root = "./data";
    }
    
    return args;
}

void run_benchmark(const ProgramArgs& args)
{
    std::cout << "=" << std::string(78, '=') << "=" << std::endl;
    std::cout << "Whisper Benchmark (C++ with ONNX)" << std::endl;
    std::cout << "=" << std::string(78, '=') << "=" << std::endl;
    std::cout << "\nBenchmark: " << args.bench_name << std::endl;
    std::cout << "Model: " << args.model_path << std::endl;
    std::cout << "Threads: " << args.num_threads << std::endl;
    std::cout << std::endl;
    
    // Create output directory
    fs::create_directories(args.output_dir);
    
    // Determine data directory path
    // Try multiple locations for backward compatibility:
    // 1. New structure: benchmark_cpp/data/golos_10_debug_extracted/golos_10_debug/
    // 2. Old structure: benchmark_cpp/data/golos/
    // 3. Custom path: --data-root parameter
    
    fs::path data_dir;
    fs::path annotation_path;
    fs::path audio_dir;
    
    // Try new ZIP-extracted structure first
    if (args.bench_name == "golos_10_debug") {
        fs::path new_path = args.data_root / "golos_10_debug_extracted" / "golos_10_debug";
        if (fs::is_directory(new_path)) {
            data_dir = new_path;
            std::cout << "Using new data structure: " << data_dir << std::endl;
        }
    }
    
    // Fallback to traditional structure
    if (data_dir.empty()) {
        data_dir = args.data_root / args.bench_name;
    }
    
    annotation_path = data_dir / "annotation.csv";
    audio_dir = data_dir / "files";
    
    std::cout << "Data directory: " << data_dir << std::endl;
    std::cout << "Audio directory: " << audio_dir << std::endl;
    
    if (!fs::is_directory(data_dir)) {
        throw std::runtime_error("Benchmark dataset not found: " + data_dir.string());
    }
    if (!fs::exists(annotation_path)) {
        throw std::runtime_error("Annotation file not found: " + annotation_path.string());
    }
    if (!fs::is_directory(audio_dir)) {
        throw std::runtime_error("Audio directory not found: " + audio_dir.string());
    }
    
    std::cout << "Loading annotation..." << std::endl;
    auto items = load_annotation(annotation_path);
    
    if (args.max_samples > 0) {
        items.resize(std::min((size_t)args.max_samples, items.size()));
    }
    
    std::cout << "Found " << items.size() << " samples" << std::endl;
    
    if (items.empty()) {
        throw std::runtime_error("No samples found in annotation");
    }
    
    // Construct paths to model files
    fs::path model_dir(args.model_path);
    if (!model_dir.parent_path().empty() && model_dir.filename() != ".") {
        // If model_path points to a specific file, use parent directory
        if (model_dir.extension() == ".onnx") {
            model_dir = model_dir.parent_path();
        }
    }
    
    std::string encoder_path = (model_dir / "encoder_model.onnx").string();
    std::string decoder_path = (model_dir / "decoder_model.onnx").string();
    std::string decoder_with_past_path = (model_dir / "decoder_with_past_model.onnx").string();
    std::string vocab_path = (model_dir / "vocab.json").string();
    
    // Load ONNX model
    std::cout << "Loading ONNX model..." << std::endl;
    WhisperONNXModel model(encoder_path, decoder_path, decoder_with_past_path, vocab_path, args.num_threads);
    
    std::cout << "Model loaded successfully\n" << std::endl;
    
    // Run benchmark
    BenchmarkStats stats;
    std::vector<TranscriptionResult> predictions;
    
    std::cout << "Running inference on " << items.size() << " samples..." << std::endl;
    
    for (size_t idx = 0; idx < items.size(); idx++) {
        const auto& [filename, reference] = items[idx];
        fs::path audio_path = audio_dir / filename;
        
        // Progress indicator
        if ((idx + 1) % 10 == 0 || idx == 0) {
            std::cout << "  [" << (idx + 1) << "/" << items.size() << "]" << std::endl;
        }
        
        TranscriptionResult result;
        result.filename = filename;
        result.reference = reference;
        
        try {
            if (!fs::exists(audio_path)) {
                result.status = "missing_file";
                result.error = "Audio file not found: " + audio_path.string();
                result.hypothesis = "";
                result.latency_ms = 0.0;
                stats.update_failure();
                predictions.push_back(result);
                continue;
            }
            
            // Load audio (supports WAV, OPUS, and other formats)
            std::cout << "    Loading: " << filename << std::flush;
            auto audio_data = AudioUtils::load_audio_file(audio_path.string(), 16000);
            std::cout << " (" << audio_data.size() << " samples)" << std::endl;
            
            // Transcribe
            auto start_time = std::chrono::high_resolution_clock::now();
            result.hypothesis = model.transcribe(audio_data, 16000);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            result.latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            // Calculate metrics
            auto [wer_dist, ref_words] = MetricsCalculator::calculate_wer(reference, result.hypothesis);
            auto [cer_dist, ref_chars] = MetricsCalculator::calculate_cer(reference, result.hypothesis);
            
            stats.update_success(result.latency_ms, wer_dist, ref_words, cer_dist, ref_chars);
            result.status = "ok";
            
        } catch (const std::exception& e) {
            result.status = "error";
            result.error = e.what();
            result.hypothesis = "";
            result.latency_ms = 0.0;
            stats.update_failure();
            std::cout << "    ERROR: " << e.what() << std::endl;
        }
        
        predictions.push_back(result);
        
        if (args.sleep_between_requests > 0) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(args.sleep_between_requests)
            );
        }
    }
    
    // Write outputs
    std::cout << "\nWriting results..." << std::endl;
    
    // Write predictions CSV
    fs::path predictions_path = args.output_dir / "predictions.csv";
    CSVWriter csv_writer(predictions_path);
    csv_writer.write_header({
        "filename", "reference", "hypothesis", "latency_ms", "status", "error"
    });
    for (const auto& pred : predictions) {
        csv_writer.write_row({
            pred.filename,
            pred.reference,
            pred.hypothesis,
            std::to_string(pred.latency_ms),
            pred.status,
            pred.error
        });
    }
    std::cout << "Predictions saved to: " << predictions_path << std::endl;
    
    // Write metrics JSON
    fs::path metrics_path = args.output_dir / "metrics.json";
    Json::Value metrics_json;
    metrics_json["bench_name"] = args.bench_name;
    metrics_json["model_path"] = args.model_path;
    metrics_json["threads"] = args.num_threads;
    metrics_json["requested_samples"] = (int)items.size();
    metrics_json["processed_samples"] = stats.processed;
    metrics_json["failed_samples"] = stats.failed;
    metrics_json["average_latency_ms"] = stats.average_latency();
    metrics_json["wer"] = stats.wer();
    metrics_json["cer"] = stats.cer();
    
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto tm_now = std::gmtime(&time_t_now);
    std::stringstream ss;
    ss << std::put_time(tm_now, "%Y-%m-%dT%H:%M:%SZ");
    metrics_json["timestamp_utc"] = ss.str();
    
    std::ofstream metrics_file(metrics_path);
    metrics_file << metrics_json.toStyledString();
    std::cout << "Metrics saved to: " << metrics_path << std::endl;
    
    // Write summary
    fs::path summary_path = args.output_dir / "summary.txt";
    std::ofstream summary_file(summary_path);
    summary_file << "Whisper Benchmark Results (C++ ONNX)\n"
                 << "====================================\n\n"
                 << "Benchmark: " << args.bench_name << "\n"
                 << "Model: " << args.model_path << "\n"
                 << "Threads: " << args.num_threads << "\n"
                 << "Samples requested: " << items.size() << "\n"
                 << "Samples processed: " << stats.processed << "\n"
                 << "Samples failed: " << stats.failed << "\n"
                 << "Average latency (ms): " << std::fixed << std::setprecision(2)
                 << stats.average_latency() << "\n"
                 << "WER: " << std::fixed << std::setprecision(4) << stats.wer() << "\n"
                 << "CER: " << std::fixed << std::setprecision(4) << stats.cer() << "\n";
    std::cout << "Summary saved to: " << summary_path << std::endl;
    
    // Print summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Results Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Samples processed: " << stats.processed << "/" << items.size() << std::endl;
    std::cout << "Samples failed: " << stats.failed << std::endl;
    std::cout << "Average latency: " << std::fixed << std::setprecision(2)
              << stats.average_latency() << " ms" << std::endl;
    std::cout << "WER: " << std::fixed << std::setprecision(4) << stats.wer() << std::endl;
    std::cout << "CER: " << std::fixed << std::setprecision(4) << stats.cer() << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

void run_single_file(const ProgramArgs& args)
{
    // Construct paths to model files
    fs::path model_dir(args.model_path);
    if (!model_dir.parent_path().empty() && model_dir.filename() != ".") {
        // If model_path points to a specific file, use parent directory
        if (model_dir.extension() == ".onnx") {
            model_dir = model_dir.parent_path();
        }
    }
    
    std::string encoder_path = (model_dir / "encoder_model.onnx").string();
    std::string decoder_path = (model_dir / "decoder_model.onnx").string();
    std::string decoder_with_past_path = (model_dir / "decoder_with_past_model.onnx").string();
    std::string vocab_path = (model_dir / "vocab.json").string();
    
    // Load ONNX model
    WhisperONNXModel model(encoder_path, decoder_path, decoder_with_past_path, vocab_path, args.num_threads);
    
    try {
        // Load audio file
        auto audio_data = AudioUtils::load_audio_file(args.single_file_path, 16000);
        
        // Transcribe
        auto start_time = std::chrono::high_resolution_clock::now();
        std::string transcription = model.transcribe(audio_data, 16000);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        double latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        if (args.output_format == "json") {
            // Output JSON format for service integration
            Json::Value result;
            result["transcription"] = transcription;
            result["latency_ms"] = latency_ms;
            result["status"] = "ok";
            
            Json::StreamWriterBuilder builder;
            builder["indentation"] = "";  // Compact output
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            writer->write(result, &std::cout);
            std::cout << std::endl;
        } else {
            // Output plain text (default)
            std::cout << transcription << std::endl;
        }
    } catch (const std::exception& e) {
        if (args.output_format == "json") {
            Json::Value result;
            result["transcription"] = "";
            result["latency_ms"] = 0.0;
            result["status"] = "error";
            result["error"] = e.what();
            
            Json::StreamWriterBuilder builder;
            builder["indentation"] = "";
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            writer->write(result, &std::cout);
            std::cout << std::endl;
        } else {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        throw;  // Re-throw to maintain exit code
    }
}

void run_interactive(const ProgramArgs& args)
{
    // Construct paths to model files
    fs::path model_dir(args.model_path);
    if (!model_dir.parent_path().empty() && model_dir.filename() != ".") {
        // If model_path points to a specific file, use parent directory
        if (model_dir.extension() == ".onnx") {
            model_dir = model_dir.parent_path();
        }
    }
    
    std::string encoder_path = (model_dir / "encoder_model.onnx").string();
    std::string decoder_path = (model_dir / "decoder_model.onnx").string();
    std::string decoder_with_past_path = (model_dir / "decoder_with_past_model.onnx").string();
    std::string vocab_path = (model_dir / "vocab.json").string();
    
    // Load model once
    std::cerr << "Loading model in interactive mode: " << args.model_path << std::endl;
    WhisperONNXModel model(encoder_path, decoder_path, decoder_with_past_path, vocab_path, args.num_threads);
    std::cerr << "Model loaded. Ready for requests." << std::endl;
    
    // Signal ready
    std::cout << "READY" << std::endl;
    std::cout.flush();
    
    std::string line;
    while (std::getline(std::cin, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        
        if (line.empty()) continue;
        
        // Check for special commands
        if (line == "PING") {
            std::cout << "PONG" << std::endl;
            std::cout.flush();
            continue;
        }
        
        if (line == "QUIT") {
            break;
        }
        
        // Check if this is a mel-spectrogram file command
        if (line.substr(0, 4) == "MEL ") {
            std::string mel_path = line.substr(4);
            
            try {
                // Read mel-spectrogram from binary file
                std::ifstream mel_file(mel_path, std::ios::binary);
                if (!mel_file.is_open()) {
                    throw std::runtime_error("Failed to open mel-spectrogram file: " + mel_path);
                }
                
                // Read shape (2 ints: n_mels, n_frames)
                int n_mels, n_frames;
                mel_file.read(reinterpret_cast<char*>(&n_mels), sizeof(int));
                mel_file.read(reinterpret_cast<char*>(&n_frames), sizeof(int));
                
                std::cerr << "Loading Python mel-spectrogram: " << n_mels << "x" << n_frames << std::endl;
                
                // Read data
                size_t data_size = n_mels * n_frames;
                std::vector<float> mel_spec(data_size);
                mel_file.read(reinterpret_cast<char*>(mel_spec.data()), data_size * sizeof(float));
                mel_file.close();
                
                // Compute statistics for debugging
                float mean = 0.0f, min_val = 1e10f, max_val = -1e10f;
                for (float val : mel_spec) {
                    mean += val;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
                mean /= mel_spec.size();
                std::cerr << "Python mel-spec: mean=" << mean << ", min=" << min_val << ", max=" << max_val << std::endl;
                
                // Run encoder with Python-generated mel-spectrogram
                auto encoder_output = model.run_encoder_from_mel(mel_spec);
                
                // Run decoder
                auto token_ids = model.run_decoder_from_encoder_output(encoder_output);
                
                // Output token IDs as JSON for Python to decode
                std::cout << "TOKENS:";
                for (size_t i = 0; i < token_ids.size(); i++) {
                    std::cout << token_ids[i];
                    if (i < token_ids.size() - 1) std::cout << ",";
                }
                std::cout << std::endl;
                std::cout.flush();
                
            } catch (const std::exception& e) {
                std::cout << "ERROR: " << e.what() << std::endl;
                std::cout.flush();
            }
            continue;
        }
        
        // Treat line as audio file path (C++ will compute mel-spectrogram)
        try {
            std::vector<float> audio_data = AudioUtils::load_audio_file(line, 16000);
            
            // Compute mel-spectrogram
            auto mel_spec = model.compute_mel_spectrogram(audio_data, 16000);
            auto encoder_output = model.run_encoder_from_mel(mel_spec);
            auto token_ids = model.run_decoder_from_encoder_output(encoder_output);
            
            // Output token IDs as JSON for Python to decode
            std::cout << "TOKENS:";
            for (size_t i = 0; i < token_ids.size(); i++) {
                std::cout << token_ids[i];
                if (i < token_ids.size() - 1) std::cout << ",";
            }
            std::cout << std::endl;
            std::cout.flush();
            
        } catch (const std::exception& e) {
            std::cout << "ERROR: " << e.what() << std::endl;
            std::cout.flush();
        }
    }
    
    std::cerr << "Interactive mode ended." << std::endl;
}

int main(int argc, char* argv[])
{
    try {
        ProgramArgs args = parse_args(argc, argv);
        
        if (args.interactive) {
            run_interactive(args);
        } else if (args.single_file_mode) {
            run_single_file(args);
        } else {
            run_benchmark(args);
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
