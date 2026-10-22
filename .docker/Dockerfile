FROM mcr.microsoft.com/devcontainers/rust:1-bookworm

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \
	curl \
	jq \
	libudev-dev \
	pkg-config \
	&& rm -rf /var/lib/apt/lists/*

USER vscode
ARG RUST_VERSION=1.94.1
RUN rustup toolchain install ${RUST_VERSION} --profile minimal \
		--component rustfmt --component llvm-tools --target thumbv6m-none-eabi \
	&& rustup default ${RUST_VERSION}
RUN cargo install flip-link --version 0.1.12 --locked \
	&& cargo install cargo-binutils --locked \
	&& cargo install elf2uf2-rs --locked

USER root
RUN chgrp -R rustlang ${CARGO_HOME} ${RUSTUP_HOME} \
	&& chmod -R g+rwX ${CARGO_HOME} ${RUSTUP_HOME}
USER vscode
