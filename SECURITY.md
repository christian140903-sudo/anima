# Security Policy

## Supported version

Security fixes are applied to the latest release on the `main` branch.

## Report a vulnerability

Please use GitHub's private vulnerability reporting for this repository rather
than opening a public issue. Include the affected version, a minimal
reproduction, the expected impact, and any suggested mitigation.

## Data-handling model

ANIMA can persist raw inputs, derived context, affect-inspired state, and event
memory to JSON files in the configured state directory. Those files are not
encrypted by ANIMA. Applications are responsible for access control,
encryption-at-rest, retention, and deletion policies appropriate to their data.

The Anthropic and OpenAI adapters send assembled context to their configured
remote APIs. The Ollama adapter sends it to the configured Ollama endpoint.
Never place secrets in event content or tags, and review provider policies
before processing sensitive data.

API keys are read from environment variables by the CLI and are not intended to
be serialized into kernel state. If you find a path that persists credentials,
please report it privately as a high-priority issue.
