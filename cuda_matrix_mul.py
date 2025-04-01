'''
A quick benchmark of a matrix multiple, comparing CUDA GPU vs CPU I7

Python Version  : 3.12.7
PyTorch Version : 2.6.0+cu126
Cuda Version    : 12.8
GPU Version     : NVIDIA GeForce RTX 4050 
CPU Version     : Intel 13th Gen i7-1362H 2.40 GHz
 
Results in seconds  (lower is better) : 

    System          CPU Time   GPU Time 
    =============+===========+=========
    i7+RTX 4050     320        0.42 
'''

import time                                                                                      	
import torch                                                                                                                                                                                          

print("Checking to see if CUDA is available")

if (torch.cuda.is_available()): # FALSE on MBP,

	    device = torch.device("cuda")                                                            

else:                                                                                                    

	device = torch.device("cpu")                                                              	

print(" device = ", device)                                                                                                                                                                           
print("Randomizing array");

matrix_size = 32 * 512;                                                                          

x = torch.randn(matrix_size, matrix_size)                                                        	

y = torch.randn(matrix_size, matrix_size)                                                                                                                                                             
print("CPU multiply");                                                                           	

start=time.time();

results = torch.matmul(x,y);

end=time.time(); 

print("CPU time=", end-start)                                                                                                                                                                                 	

print("GPU multiply");                                                                           	

x_gpu = x.to(device)                                                                             	

y_gpu = y.to(device)                                                                             	

start=time.time();

results_gpu = torch.matmul(x_gpu,y_gpu);

end=time.time(); 

print("GPU time=", end-start)                                                                                     	

