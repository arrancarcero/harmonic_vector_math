@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.29
nvcc --shared harmonic_reduction.cu -o harmonic_reduction.dll
nvcc --shared harmonic_cutile_stride.cu -o harmonic_stride.dll
