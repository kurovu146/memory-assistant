# syntax=docker/dockerfile:1.6

FROM rust:1-bookworm AS builder
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
    && echo "fn main() {}" > src/main.rs \
    && echo "" > src/lib.rs \
    && cargo build --release \
    && rm -rf src target/release/deps/memory_assistant* target/release/memory-assistant*

COPY src ./src
RUN touch src/main.rs src/lib.rs \
    && cargo build --release --bin memory-assistant

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /data
COPY --from=builder /app/target/release/memory-assistant /usr/local/bin/memory-assistant

CMD ["memory-assistant"]
