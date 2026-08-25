.PHONY: install run ping status export-continue test clean

install:
	pip install -e mesh_gateway

run:
	agent-mesh run

ping:
	agent-mesh ping

status:
	agent-mesh status

export-continue:
	agent-mesh export-continue

test:
	pytest -v tests/

clean:
	rm -rf __pycache__ .pytest_cache dist build *.egg-info
