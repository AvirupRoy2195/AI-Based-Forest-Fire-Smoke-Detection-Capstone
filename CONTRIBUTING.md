<![CDATA[# 🤝 Contributing to AI-Based Forest Fire & Smoke Detection

Thank you for your interest in contributing to this project! This document provides guidelines and best practices for contributing.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

---

## 📜 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Accept responsibility for mistakes and learn from them

---

## 🚀 Getting Started

### 1. Fork the Repository

Click the "Fork" button on the [main repository page](https://github.com/AvirupRoy2195/AI-Based-Forest-Fire-Smoke-Detection-Capstone).

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/AI-Based-Forest-Fire-Smoke-Detection-Capstone.git
cd AI-Based-Forest-Fire-Smoke-Detection-Capstone
```

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv fire_detection_env

# Activate (Windows)
fire_detection_env\Scripts\activate

# Activate (macOS/Linux)
source fire_detection_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Add Upstream Remote

```bash
git remote add upstream https://github.com/AvirupRoy2195/AI-Based-Forest-Fire-Smoke-Detection-Capstone.git
```

---

## 🔄 Development Workflow

### 1. Sync with Upstream

Before starting work, ensure your fork is up to date:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

**Branch Naming Conventions:**

| Type | Format | Example |
|------|--------|---------|
| Feature | `feature/description` | `feature/add-svm-model` |
| Bug Fix | `fix/description` | `fix/scaling-issue` |
| Documentation | `docs/description` | `docs/update-readme` |
| Refactor | `refactor/description` | `refactor/optimize-pipeline` |

### 3. Make Changes

- Write clean, documented code
- Follow the coding standards below
- Test your changes locally

### 4. Commit Changes

```bash
git add .
git commit -m "feat: Add your descriptive message"
```

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create Pull Request

Open a pull request from your fork to the upstream `main` branch.

---

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with the following specifics:

```python
# ✅ Good: Descriptive function names with docstrings
def calculate_spectral_ratio(red_channel, green_channel, epsilon=1e-10):
    """
    Calculate the spectral ratio between red and green channels.

    Parameters
    ----------
    red_channel : array-like
        Red channel values
    green_channel : array-like
        Green channel values
    epsilon : float, optional
        Small value to prevent division by zero (default: 1e-10)

    Returns
    -------
    array-like
        Spectral ratio values

    Examples
    --------
    >>> ratio = calculate_spectral_ratio([100, 150], [80, 120])
    """
    return red_channel / (green_channel + epsilon)

# ❌ Bad: Unclear naming, no documentation
def calc(r, g):
    return r / g
```

### Jupyter Notebook Guidelines

1. **Cell Organization**
   - Use markdown cells to document sections
   - Keep code cells focused and concise
   - Clear outputs before committing

2. **Documentation**
   ```python
   # =========================================
   # Section Title
   # =========================================
   # Description of what this section does
   ```

3. **Reproducibility**
   ```python
   # Always set random seeds at the start
   SEED = 42
   np.random.seed(SEED)
   random.seed(SEED)
   ```

### Import Organization

```python
# Standard library imports
import os
import sys
import json

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports (if any)
from utils import helper_function
```

---

## 📌 Commit Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Commit Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, semicolons, etc.) |
| `refactor` | Code refactoring |
| `perf` | Performance improvement |
| `test` | Adding/modifying tests |
| `chore` | Maintenance tasks |

### Examples

```bash
# Feature
git commit -m "feat: Add LightGBM model to comparison pipeline"

# Bug fix
git commit -m "fix: Correct feature scaling in preprocessing"

# Documentation
git commit -m "docs: Update installation instructions in README"

# Refactor
git commit -m "refactor: Optimize feature selection function"
```

---

## 🔍 Pull Request Process

### 1. PR Title Format

Follow the same convention as commits:
```
feat: Add ensemble voting classifier
```

### 2. PR Description Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Changes Made
- Change 1
- Change 2

## Testing
Describe how you tested your changes.

## Screenshots (if applicable)
Add screenshots for UI changes.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed my code
- [ ] Added/updated documentation
- [ ] Changes don't break existing functionality
```

### 3. Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all review comments
4. Squash commits if requested

---

## 🐛 Issue Reporting

### Bug Reports

Use this template:

```markdown
## Bug Description
A clear description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Windows 11]
- Python Version: [e.g., 3.9.7]
- Package Versions: [from pip freeze]

## Additional Context
Screenshots, error logs, etc.
```

### Feature Requests

```markdown
## Feature Description
A clear description of the proposed feature.

## Motivation
Why is this feature needed?

## Proposed Solution
How would you implement this?

## Alternatives Considered
Other approaches you've thought about.
```

---

## 🎯 Areas for Contribution

### High Priority

- [ ] Add more ML models (SVM, Neural Networks)
- [ ] Improve hyperparameter tuning
- [ ] Add unit tests
- [ ] Create Docker containerization

### Medium Priority

- [ ] Add data augmentation
- [ ] Implement ensemble methods
- [ ] Create REST API for predictions
- [ ] Add more visualization options

### Low Priority / Nice to Have

- [ ] Add multi-language documentation
- [ ] Create interactive Streamlit app
- [ ] Add CI/CD pipeline

---

## 📞 Getting Help

If you need help:

1. Check existing [Issues](https://github.com/AvirupRoy2195/AI-Based-Forest-Fire-Smoke-Detection-Capstone/issues)
2. Read the documentation
3. Open a new issue with your question

---

## 🙏 Thank You!

Your contributions help improve forest fire detection and prevention. Every contribution, no matter how small, is valued!

---

*Happy Coding! 🔥🌲*
]]>
