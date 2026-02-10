# Contributing to J.A.R.V.I.S.

First off — thank you for considering contributing. Whether it's a bug report, feature request, or a pull request, every contribution helps make the Jetson AI community stronger.

## Quick Links

- [Issues](https://github.com/steffenpharai/Jarvis/issues) — report bugs or request features
- [Discussions](https://github.com/steffenpharai/Jarvis/discussions) — ask questions, share your setup
- [Roadmap](../README.md#-roadmap) — see what's planned

## Development Setup

```bash
# Clone
git clone https://github.com/steffenpharai/Jarvis.git && cd Jarvis

# Python environment
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# Download models
bash scripts/bootstrap_models.sh

# PWA frontend
cd pwa && npm install && npm run build && cd ..

# Verify everything works
ruff check .
pytest tests/unit/
python main.py --dry-run
```

## Making Changes

1. **Fork** the repo and create a branch from `main`
2. **Write code** — follow the existing style (ruff handles formatting)
3. **Add tests** — new features need tests in `tests/unit/` or `tests/e2e/`
4. **Run the checks**:

```bash
ruff check .              # Lint
pytest tests/unit/        # Unit tests (no hardware needed)
python main.py --dry-run  # Smoke test
```

5. **Submit a PR** with a clear description of what and why

## Code Style

- **Python 3.10+** — we use modern syntax (match/case, type hints, etc.)
- **Ruff** for linting — `ruff check .` must pass with zero errors
- **Line length** — 100 chars (configured in `pyproject.toml`)
- **Docstrings** — Google style for public functions
- **Commit messages** — imperative mood, concise ("Add vision depth fallback", not "Added stuff")

## What We're Looking For

### High-impact contributions
- VLM integration (LLaVA, Qwen-VL) for native image understanding
- ROS 2 bridge for robotics integration
- Docker image for JetPack 6.x
- Home Assistant integration
- Multi-camera / multi-room support
- Performance optimizations on 8GB RAM

### Always welcome
- Bug fixes with test coverage
- Documentation improvements
- New E2E tests for hardware scenarios
- PWA UI/UX improvements
- New Piper voice models

### Hardware testing
If you have a Jetson Orin Nano, running the E2E tests and reporting results is incredibly valuable:

```bash
pytest tests/e2e/ -m e2e -v 2>&1 | tee e2e_results.txt
```

Share results in an issue or discussion!

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include before/after benchmarks for performance changes
- Update README.md if adding user-facing features
- All CI checks must pass (lint + unit tests)
- E2E tests are run manually on hardware — note if your change affects them

## Questions?

Open a [Discussion](https://github.com/steffenpharai/Jarvis/discussions). No question is too basic.

---

*"I'm not saying I'll merge it immediately, sir, but I'll certainly give it my full attention."*
