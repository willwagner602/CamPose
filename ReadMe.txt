To Run:
1. Extract images to ./images
2. Ensure pattern name is "pattern.jpg"
3. Ensure image names contain "IMG"
4. Manually install numpy-1.12.0rc1+mkl because it's not available via pip
	a. On Windows - http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
4. pip install -r requirements.txt
	a. Scipy may fail to build on windows, use http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
5. Run CameraPose.py

Tested and works on Windows 10 with Python 3.5