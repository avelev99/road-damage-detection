# Contribution Guidelines

Welcome! We appreciate your interest in improving this road damage detection system. Please follow these guidelines to ensure smooth collaboration.

## Code of Conduct
This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/). By participating, you agree to uphold this code.

## Reporting Bugs
1. Check existing issues to avoid duplicates
2. Use the bug report template:
   ```markdown
   ### Describe the bug
   [Clear description]

   ### To Reproduce
   Steps to reproduce:
   1. ...
   2. ...

   ### Expected behavior
   [What should happen]

   ### Screenshots/Logs
   [If applicable]

   ### Environment
   - OS: [e.g. Ubuntu 22.04]
   - Hardware: [e.g. Jetson Nano]
   - Python version: [e.g. 3.9]
   ```
   
## Suggesting Enhancements
1. Check existing feature requests
2. Use the enhancement template:
   ```markdown
   ### Is your feature request related to a problem?
   [Description]

   ### Describe the solution
   [Proposed solution]

   ### Describe alternatives
   [Alternative solutions]

   ### Additional context
   [Screenshots, research papers, etc.]
   ```

## Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/road-damage-detection.git
cd road-damage-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

## Pull Request Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

## Coding Standards
- Follow PEP 8 for Python code
- Use type hints for all function signatures
- Document public methods with docstrings
- Keep functions under 50 lines
- Write unit tests for new functionality