# Brain Tumor Classification

A deep learning application for classifying brain tumors using MRI images.

## GitHub Actions Status
[![Docker Build and Test](https://github.com/{username}/{repository}/actions/workflows/docker-build.yml/badge.svg)](https://github.com/{username}/{repository}/actions/workflows/docker-build.yml)

## Overview
This project uses Streamlit and MLflow to serve a deep learning model that classifies brain MRI images into four categories:
- Glioma
- Meningioma
- No Tumor
- Pituitary

## CI/CD Pipeline
The project uses GitHub Actions for continuous integration and deployment:

1. **Automated Testing**: On every push and pull request to main/master branches:
   - Builds Docker containers
   - Runs integration tests
   - Checks service health
   - Verifies model availability

2. **Container Build**:
   - Builds Streamlit application container
   - Builds MLflow tracking server container
   - Tests container connectivity

## Local Development
1. Clone the repository:
   ```bash
   git clone https://github.com/{username}/{repository}.git
   cd {repository}
   ```

2. Start the services:
   ```bash
   docker compose up --build
   ```

3. Access the applications:
   - Streamlit: http://localhost:8999
   - MLflow: http://localhost:5000

## Running Tests
```bash
pip install pytest requests
pytest tests/
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Add your license here] 