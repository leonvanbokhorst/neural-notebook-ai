# Browser Automation POC

This POC demonstrates browser automation capabilities using both Selenium and Playwright frameworks.

## Features

- Multi-browser framework support (Selenium and Playwright)
- Headless mode support
- Screenshot capture
- Configurable wait timeouts
- Error handling and resource cleanup
- Logging and monitoring

## Requirements

- Python 3.8+
- Chrome/Chromium browser installed
- Dependencies listed in `requirements.txt`

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install browser drivers:

```bash
# For Playwright
playwright install chromium

# For Selenium (ChromeDriver)
# ChromeDriver installation is automatically managed by Selenium 4+
```

## Configuration

Edit `config.yaml` to customize:

- Target URL
- Browser framework (selenium/playwright)
- Headless mode
- Screenshot directory
- Wait timeouts

## Usage

Run the demonstration:

```bash
python browser_use_demo.py
```

## Output

The POC will:

1. Navigate to the configured URL
2. Capture a screenshot
3. Save browser metrics
4. Clean up resources

Screenshots are saved in the configured directory (`screenshots/` by default).

## Development

- Format code: `black .`
- Type checking: `mypy .`
- Linting: `flake8`
- Run tests: `pytest`
