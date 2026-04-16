#!/bin/bash
# =============================================================================
# OptiS Benchmark - Data Download Script
# =============================================================================

set -e

# Configuration
DATASET_DIR="dataset"
PROCESSED_DIR="${DATASET_DIR}/processed"
RAW_DIR="${DATASET_DIR}/raw"
GITHUB_REPO="${GITHUB_REPO:-your-org/optis_benchmark}"
GITHUB_URL="https://github.com/${GITHUB_REPO}/releases/download"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check for curl or wget
    if command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl -L -o"
    elif command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget -O"
    else
        error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        error "git not found. Please install git."
        exit 1
    fi
    
    info "Prerequisites check passed."
}

# Create directories
create_directories() {
    info "Creating directories..."
    mkdir -p "${PROCESSED_DIR}"
    mkdir -p "${RAW_DIR}"
    mkdir -p "${DATASET_DIR}/external"
    info "Directories created."
}

# Download dataset
download_dataset() {
    local dataset_name=$1
    local version=$2
    local filename="${dataset_name}.tar.gz"
    
    info "Downloading ${dataset_name}..."
    
    if command -v curl &> /dev/null; then
        curl -L -o "${RAW_DIR}/${filename}" \
            "${GITHUB_URL}/${version}/${filename}"
    else
        wget -O "${RAW_DIR}/${filename}" \
            "${GITHUB_URL}/${version}/${filename}"
    fi
    
    if [ $? -eq 0 ]; then
        info "${dataset_name} downloaded successfully."
    else
        error "Failed to download ${dataset_name}."
        return 1
    fi
}

# Extract dataset
extract_dataset() {
    local filename=$1
    
    info "Extracting ${filename}..."
    
    tar -xzf "${RAW_DIR}/${filename}" -C "${PROCESSED_DIR}"
    
    if [ $? -eq 0 ]; then
        info "${filename} extracted successfully."
    else
        error "Failed to extract ${filename}."
        return 1
    fi
}

# Download all datasets
download_all() {
    info "Downloading all datasets..."
    
    local version="v1.0.0"
    
    # Download lens design dataset
    download_dataset "lens_design" "${version}"
    extract_dataset "lens_design.tar.gz"
    
    # Download system analysis dataset
    download_dataset "system_analysis" "${version}"
    extract_dataset "system_analysis.tar.gz"
    
    # Download tolerance analysis dataset
    download_dataset "tolerance_analysis" "${version}"
    extract_dataset "tolerance_analysis.tar.gz"
    
    info "All datasets downloaded and extracted."
}

# Download sample dataset (for testing)
download_sample() {
    info "Downloading sample dataset..."
    
    local sample_file="sample_data.jsonl"
    local url="https://raw.githubusercontent.com/${GITHUB_REPO}/main/dataset/processed/${sample_file}"
    
    if command -v curl &> /dev/null; then
        curl -L -o "${PROCESSED_DIR}/${sample_file}" "${url}"
    else
        wget -O "${PROCESSED_DIR}/${sample_file}" "${url}"
    fi
    
    if [ $? -eq 0 ]; then
        info "Sample dataset downloaded successfully."
    else
        warn "Failed to download sample dataset. This is optional."
    fi
}

# Verify downloads
verify_downloads() {
    info "Verifying downloads..."
    
    local count=$(find "${PROCESSED_DIR}" -name "*.jsonl" | wc -l)
    
    if [ ${count} -gt 0 ]; then
        info "Found ${count} dataset file(s)."
    else
        warn "No dataset files found in ${PROCESSED_DIR}."
    fi
}

# Cleanup
cleanup() {
    info "Cleaning up temporary files..."
    rm -f "${RAW_DIR}"/*.tar.gz
    info "Cleanup complete."
}

# Main
main() {
    echo "========================================"
    echo "OptiS Benchmark - Dataset Downloader"
    echo "========================================"
    echo ""
    
    check_prerequisites
    create_directories
    
    # Check for --sample flag
    if [ "$1" == "--sample" ]; then
        download_sample
    else
        download_all
    fi
    
    verify_downloads
    
    echo ""
    echo "========================================"
    info "Dataset preparation complete!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review downloaded datasets: ls ${PROCESSED_DIR}"
    echo "  2. Run evaluation: python src/main.py --agent-config configs/agents/gpt-4.yaml --task-set lens_design"
    echo ""
}

# Run
main "$@"
