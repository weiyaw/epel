FROM python:3.13

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

WORKDIR /home
