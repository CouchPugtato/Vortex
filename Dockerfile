# Multi-stage build for dumapril-taglocalization
FROM rust:1.82 AS builder
WORKDIR /app

# Cache deps
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo "fn main(){}" > src/main.rs
RUN cargo build --release || true

# Copy source
COPY src ./src
COPY .cargo ./.cargo
RUN cargo build --release

FROM debian:bookworm-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/dumapril-taglocalization /usr/local/bin/dumapril-taglocalization

# Default envs (can be overridden at runtime)
ENV APRILTAG_STATIC=1

ENTRYPOINT ["/usr/local/bin/dumapril-taglocalization"]
CMD ["/data/input", "/data/output"]