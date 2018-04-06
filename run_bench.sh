echo "Compiling and running simple vector addition.."
nvcc -o vec_add vecadd.cu -std=c++11;./vec_add

echo "Running naive matrix doublify operation via pycuda.."
python naive_doublify.py
