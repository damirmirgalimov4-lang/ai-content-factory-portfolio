# Security policy

## Supported version

Security fixes are applied to the current `main` branch of this portfolio snapshot.

## Reporting a vulnerability

Please use GitHub's **Report a vulnerability** flow under the repository Security tab. Do not publish credentials, private data, exploit details or provider task identifiers in a public issue.

If you find a credential in any revision:

1. do not use or validate it;
2. identify only the file and line through the private advisory;
3. treat the value as compromised and rotate it at the provider;
4. remove it from both the current tree and Git history before the next release.

## Scope

The repository contains no hosted service and no production credentials. Provider accounts, model endpoints and private deployments are outside this public snapshot's vulnerability-reward scope.

Technical controls and known limitations are documented in [docs/SECURITY.md](docs/SECURITY.md).
