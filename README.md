# cudabench

A small set of CUDA benchmarks I am writing while I learn CUDA. In C++11 and pyCUDA. 

## Running all benchmarks

```
./run_bench.sh
```

## Running the vector addition benchmark

```
nvcc -o vec_add vecadd.cu -std=c++11;./vec_add
```

## Running the naive doublify benchmark

```
python naive_doublify.py
```
