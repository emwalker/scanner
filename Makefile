lint:
	cargo fmt
	cargo clippy --fix --allow-dirty -- -D clippy::print_stdout -D clippy::print_stderr
