# Brain Tumor Classification

A deep learning application for classifying brain tumors using MRI images.

## GitHub Actions Status
[![Python Application](https://github.com/{username}/{repository}/actions/workflows/python-app.yml/badge.svg)](https://github.com/{username}/{repository}/actions/workflows/python-app.yml)

## Overview
This project uses Streamlit and MLflow to serve a deep learning model that classifies brain MRI images into four categories:
- Glioma
- Meningioma
- No Tumor
- Pituitary

## Project Structure
```
.
├── data1/               # Training and testing data
├── pipelines/           # Training pipeline modules
├── src/                 # Source code
├── tests/              # Test files
├── .github/            # GitHub Actions workflows
├── mlruns/             # MLflow tracking files
├── .streamlit/         # Streamlit configuration
├── run_pipeline.py     # Main pipeline execution script
└── requirements.txt    # Python dependencies
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/{username}/{repository}.git
   cd {repository}
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the services:
   ```bash
   docker compose up --build
   ```

4. Access the applications:
   - Streamlit: http://localhost:8999
   - MLflow: http://localhost:5000

## Running the Pipeline
```bash
python run_pipeline.py
```

## Running Tests
```bash
pytest tests/
```

## CI/CD Pipeline
The project uses GitHub Actions for continuous integration and deployment:

1. **Automated Testing**: On every push and pull request to main branch:
   - Runs Python tests
   - Performs code linting
   - Verifies dependencies

2. **Quality Checks**:
   - Flake8 for code style
   - Pytest for unit testing

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[MIT License](LICENSE) or specify your chosen license

## Contact
Add your contact information or ways to reach out for questions and contributions. 