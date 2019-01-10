@echo off

set anaconda=C:\ProgramData\Anaconda3\python.exe
set py=E:\git\pca\pca_byPython.py
set input=E:\git\TFRecord_example\in\noise\test\filename.txt
set dirout=E:\git\pca\output
set num_case=500

call %anaconda% %py% -i1 %dirout% -i2 %input% -i3 %num_case%

pause