CC = clang
CCFLAGS = -g -fopenmp -O3
OUTPUT = *.out *.bc *.o

3D: 3D_main.c 3D_module.c
	$(CC) $(CCFLAGS) -o 3D 3D_main.c 3D_module.c -lm

old_3D:
	$(CC) -o old_3D $(CCFLAGS) 3D.c -lm 

clean:
	rm -f 3D $(OUTPUT)
