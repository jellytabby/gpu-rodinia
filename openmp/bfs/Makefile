# C compiler
CC = clang++
ICC = icc
CC_FLAGS = -fopenmp -O3 -DOPEN

all: bfs 

bfs: bfs_main.cpp bfs_module.cpp
	$(CC) $(CC_FLAGS) bfs_main.cpp bfs_module.cpp -o bfs 

old_bfs: bfs.cpp
	$(CC) $(CC_FLAGS) bfs.cpp -o old_bfs

bfs_offload: bfs.cpp
	$(ICC) $(CC_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload

clean:
	rm -f bfs bfs_offload result.txt *.o *.bc
