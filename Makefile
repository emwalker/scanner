lint:
	cargo check --all
	cargo fmt --all
	cargo clippy --fix --allow-dirty -- -D clippy::print_stdout -D clippy::print_stderr -D clippy::cognitive_complexity -D warnings

test:
	@echo "Running standard tests..."
	cargo test --quiet
	@echo ""
	@echo "Running Loom concurrency tests..."
	RUSTFLAGS="--cfg loom" cargo test --test loom_shutdown_test --release --quiet
