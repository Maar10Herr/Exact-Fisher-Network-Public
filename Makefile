.PHONY: test test-python test-cli cli check-r clean

test: test-python test-cli

test-python:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m unittest discover -s tests -v

test-cli:
	$(MAKE) -C cli clean test
	./cli/fisher-network --help >/dev/null

cli:
	$(MAKE) -C cli all

check-r:
	R CMD build r-package/fexactnet
	R CMD check --no-manual --as-cran fexactnet_0.2.0.tar.gz

clean:
	$(MAKE) -C cli clean
