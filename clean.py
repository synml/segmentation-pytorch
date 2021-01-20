import shutil

shutil.rmtree('demo', ignore_errors=True)
shutil.rmtree('result', ignore_errors=True)
shutil.rmtree('runs', ignore_errors=True)
shutil.rmtree('weights', ignore_errors=True)
shutil.rmtree('__pycache__', ignore_errors=True)
shutil.rmtree('models/__pycache__', ignore_errors=True)
