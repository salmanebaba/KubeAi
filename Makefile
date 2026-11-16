.PHONY: test demo router
test:
	python3 -m unittest discover -s tests -v
	node --check kubeai/static/app.js
demo:
	python3 -m kubeai demo
router:
	python3 -m kubeai router
