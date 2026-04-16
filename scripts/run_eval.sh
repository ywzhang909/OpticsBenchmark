#!/bin/bash
# =============================================================================
# OptiS Benchmark - Evaluation Runner Script
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
SRC_DIR="${PROJECT_ROOT}/src"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Usage
usage() {
    header "OptiS Benchmark - Evaluation Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -a, --agent <config>       Agent configuration file (required)"
    echo "  -t, --task <name>         Task set name (required)"
    echo "  -o, --output <path>        Output path (default: results/output.jsonl)"
    echo "  -c, --concurrency <n>     Max parallel tasks (default: 1)"
    echo "  --timeout <seconds>        Task timeout (default: 300)"
    echo "  -v, --verbose              Verbose output"
    echo "  -h, --help                 Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 -a configs/agents/gpt-4.yaml -t lens_design"
    echo "  $0 -a configs/agents/claude-3.yaml -t system_analysis -c 4 -v"
    echo ""
}

# Parse arguments
AGENT_CONFIG=""
TASK_SET=""
OUTPUT="results/output.jsonl"
CONCURRENCY=1
TIMEOUT=300
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--agent)
            AGENT_CONFIG="$2"
            shift 2
            ;;
        -t|--task)
            TASK_SET="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "${AGENT_CONFIG}" ]; then
    error "Agent configuration is required. Use -a or --agent."
    usage
    exit 1
fi

if [ -z "${TASK_SET}" ]; then
    error "Task set is required. Use -t or --task."
    usage
    exit 1
fi

# Change to project root
cd "${PROJECT_ROOT}"

# Check Python environment
info "Checking Python environment..."
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    error "Python not found. Please install Python 3.10+."
    exit 1
fi

PYTHON_CMD="${PYTHON:-python3}"

# Check if dependencies are installed
info "Checking dependencies..."
if ! ${PYTHON_CMD} -c "import openai" &> /dev/null; then
    warn "Dependencies not installed. Installing..."
    ${PYTHON_CMD} -m pip install -r requirements.txt
fi

# Create results directory
mkdir -p "$(dirname "${OUTPUT}")"

# Build command
CMD="${PYTHON_CMD} ${SRC_DIR}/main.py"
CMD="${CMD} --agent-config ${AGENT_CONFIG}"
CMD="${CMD} --task-set ${TASK_SET}"
CMD="${CMD} --output ${OUTPUT}"
CMD="${CMD} --concurrency ${CONCURRENCY}"
CMD="${CMD} --timeout ${TIMEOUT}"

if [ "${VERBOSE}" = true ]; then
    CMD="${CMD} --verbose"
fi

# Run
header "Starting Evaluation"
echo ""
info "Agent:       ${AGENT_CONFIG}"
info "Task:        ${TASK_SET}"
info "Output:      ${OUTPUT}"
info "Concurrency: ${CONCURRENCY}"
info "Timeout:     ${TIMEOUT}s"
echo ""
info "Command: ${CMD}"
echo ""

eval ${CMD}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    header "Evaluation Complete!"
    info "Results saved to: ${OUTPUT}"
else
    error "Evaluation failed with exit code: ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
