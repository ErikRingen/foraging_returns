# Test suite for foraging returns model refactoring

This directory contains tests that define the expected behavior of the refactored codebase.
These tests are intentionally written to fail initially - they will only pass when the corresponding
features are correctly implemented.

## Test Organization

- `test_setup.py`: Tests for package management, dependencies, and project structure
- `test_preprocessing.py`: Tests for preprocessing functions
- `test_foraging_data.py`: Tests for ForagingData class
- `test_foraging_model.py`: Tests for ForagingModel class
- `test_dimensions.py`: Tests for dimension handling and broadcasting
- `conftest.py`: Shared fixtures and test utilities

## Test Philosophy

These tests follow test-driven development principles:

1. **Tests define expected behavior**: Each test specifies what the code should do
2. **Tests fail initially**: They are written before implementation
3. **Tests should never be changed to "cheat"**: If a test fails, fix the code, not the test
4. **Tests are intentional**: Each test has a clear purpose and documents expected behavior

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_preprocessing.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=foraging_model --cov=preprocessing
```

## Test Status

- [ ] `test_setup.py`: Package management and dependencies
- [ ] `test_preprocessing.py`: Preprocessing functions
- [ ] `test_foraging_data.py`: ForagingData class
- [ ] `test_foraging_model.py`: ForagingModel class (current structure)
- [ ] `test_foraging_model.py`: ForagingModel class (dims module)
- [ ] `test_foraging_model.py`: ForagingModel class (new structure)
- [ ] `test_dimensions.py`: Dimension handling

## Notes

- Tests may skip if dependencies or data files are not available
- Some tests are placeholders that will be refined as implementation progresses
- Tests are designed to be run in parallel where possible

