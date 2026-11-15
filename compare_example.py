"""Small Python entry point for comparing ASR models from your own script."""
from asr_comparison import run_comparison

if __name__ == "__main__":
    result = run_comparison(
        manifest_path="asr_manifest_template.csv",
        output_dir="outputs/asr_comparison",
        whisper_model="base",
    )
    print(result)
