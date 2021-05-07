import os
import shutil

shutil.rmtree('demo', ignore_errors=True)
shutil.rmtree('feature_maps', ignore_errors=True)
shutil.rmtree('result', ignore_errors=True)
shutil.rmtree('runs', ignore_errors=True)
shutil.rmtree('__pycache__', ignore_errors=True)
shutil.rmtree('datasets/__pycache__', ignore_errors=True)
shutil.rmtree('models/backbone/__pycache__', ignore_errors=True)
shutil.rmtree('utils/__pycache__', ignore_errors=True)
os.system('clear')
