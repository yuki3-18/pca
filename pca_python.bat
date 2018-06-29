@echo off

set anaconda=C:\Users\LUZHIHUI\Anaconda3\python.exe
set py=F:\python_program\pca_python\pca_byPython.py
set input=F:\PCA_python_debug\input.txt
set dirout=F:\PCA_python_debug\test
set num_case=500

call %anaconda% %py% -i1 %dirout% -i2 %input% -i3 %num_case%

pause