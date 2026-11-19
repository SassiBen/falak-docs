FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd --create-home docs
USER docs
ENV FALAKDOCS_INDEX=/data/index
EXPOSE 8002
CMD ["falak-docs", "serve"]
