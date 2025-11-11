lint:
	cargo check --all --all-targets
	cargo +nightly fmt --all
	cargo clippy --allow-dirty --fix --all-targets --all-features --allow-staged -- -W clippy::all
	cargo clippy -- -D clippy::print_stdout -D clippy::print_stderr -D clippy::cognitive_complexity -D warnings

test:
	@echo "Running standard tests..."
	cargo test --quiet
	@echo ""
	@echo "Running Loom concurrency tests..."
	RUSTFLAGS="--cfg loom" cargo test --test shutdown --release --quiet --no-default-features --features soapysdr
