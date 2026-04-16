# Contributing to OptiS Benchmark

Thank you for your interest in contributing to OptiS Benchmark! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Ways to Contribute](#ways-to-contribute)
4. [Development Workflow](#development-workflow)
5. [Adding New Tasks](#adding-new-tasks)
6. [Code Style](#code-style)
7. [Testing](#testing)
8. [Submitting Changes](#submitting-changes)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- (Optional) Conda or venv for environment management

### Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/optis_benchmark.git
   cd optis_benchmark
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install development dependencies:
   ```bash
   pip install pytest pytest-asyncio black isort ruff mypy
   ```

6. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## 🤝 Ways to Contribute

### 🐛 Reporting Bugs

Before creating a bug report:
- Search existing issues
- Check if the issue is already reported

When filing a bug report, include:
- A clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, and relevant packages
- Error messages and stack traces

### 💡 Suggesting Features

We welcome feature suggestions! Please:
- Search existing issues first
- Provide a clear description
- Explain the use case and motivation
- Include mockups or examples if applicable

### 📝 Contributing Code

We accept contributions in the following areas:
- Bug fixes
- New features
- Documentation improvements
- New evaluation tasks
- Dataset contributions

## 🔄 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `task/` - New evaluation tasks
- `dataset/` - Dataset additions

### 2. Make Changes

1. Write your code
2. Follow our code style (see below)
3. Add tests
4. Update documentation

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agent.py

# Run with coverage
pytest --cov=src tests/
```

### 4. Commit Changes

Follow the commit message format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat(tasks): add tolerance analysis task

Add a new task type for evaluating agent ability to perform
tolerance analysis on optical systems.

- Add tolerance_analysis task config
- Add evaluation metrics for sensitivity
- Add sample dataset with 50 instances

Closes #123
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 📚 Adding New Tasks

### 1. Prepare Data

Create a JSONL file with task instances:

```json
{"task_id": "task_001", "instruction": "...", "expected_output": {...}, "metadata": {...}}
{"task_id": "task_002", "instruction": "...", "expected_output": {...}, "metadata": {...}}
```

### 2. Create Task Configuration

Create `configs/tasks/your_task.yaml`:

```yaml
task:
  id: "your_task"
  name: "Your Task Name"
  description: "Description of the task"
  category: "your_category"
  difficulty: 2

dataset:
  path: "dataset/processed/your_task.jsonl"
  num_samples: 50

evaluation:
  scoring_method: "metric_based"
  metrics:
    - name: "accuracy"
      type: "numeric"
  success_criteria:
    - metric: "accuracy"
      operator: ">="
      value: 0.8
```

### 3. Add Prompt Template

Create `prompts/templates/your_task.txt` with Mustache-style templates.

### 4. Submit

Follow the standard contribution workflow.

## 🎨 Code Style

### Python

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints
- Docstrings for all public functions

Format your code:
```bash
black .
isort .
```

Check:
```bash
ruff check .
mypy src/
```

## 🧪 Testing

### Writing Tests

```python
import pytest

def test_my_function():
    assert my_function(2, 3) == 5
```

### Test File Naming

- Unit tests: `tests/test_<module>.py`
- Integration tests: `tests/integration/test_<feature>.py`

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific file
pytest tests/test_agent.py -v

# Specific test
pytest tests/test_agent.py::test_chat -v
```

## 📤 Submitting Changes

### Pull Request Checklist

- [ ] My code follows the code style
- [ ] I have added tests
- [ ] All tests pass
- [ ] I have updated documentation
- [ ] My commits are properly formatted
- [ ] I have linked related issues

### PR Template

```markdown
## Summary
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2

## Testing
How was this tested?

## Screenshots (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
```

## 📞 Questions?

Feel free to:
- Open an issue for questions
- Join our discussions
- Email the maintainers

Thank you for contributing! 🎉
