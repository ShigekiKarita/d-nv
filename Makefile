test:
	dub test -b=unittest-cov --compiler=dmd

coverage: test
	@find . -name "source-*.lst" -print | xargs -I{} sh -c \
	"grep -A1 -B1 -n --color=auto 0000000 {}; tail -n 1 {}"

coveralls: test
	dub run doveralls -- -t ms4TIpE3i9sXyM9JYivIPSM5BjdY957Ob
