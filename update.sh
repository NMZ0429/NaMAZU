rm -rf build dist NaMAZU.egg-info
python setup.py bdist_wheel
twine upload dist/*
