lint:
	cargo check --all --all-targets
	cargo fmt --all
	cargo clippy --allow-dirty --fix --all-targets --all-features --allow-staged -- -W clippy::all
	cargo clippy -- -D clippy::print_stdout -D clippy::print_stderr -D clippy::cognitive_complexity -D warnings

test:
	@echo "Running standard tests..."
	cargo test --quiet
	@echo ""
	@echo "Running Loom concurrency tests..."
	RUSTFLAGS="--cfg loom" cargo test --test loom_shutdown_test --release --quiet
