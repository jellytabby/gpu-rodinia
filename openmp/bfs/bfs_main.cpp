#include "bfs_module.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #define NUM_THREAD 4

extern "C" void __mc_profiling_begin(void);
extern "C" void __mc_profiling_end(void);

FILE *fp;

void BFSGraph(int argc, char **argv);

void Usage(int argc, char **argv) {

  fprintf(stderr, "Usage: %s <num_threads> <input_file>\n", argv[0]);
}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { BFSGraph(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char **argv) {
  int no_of_nodes = 0;
  int edge_list_size = 0;
  char *input_f;
  int num_omp_threads;

  if (argc != 3) {
    Usage(argc, argv);
    exit(0);
  }

  num_omp_threads = atoi(argv[1]);
  input_f = argv[2];

  printf("Reading File\n");
  // Read in Graph from a file
  fp = fopen(input_f, "r");
  if (!fp) {
    printf("Error Reading graph file\n");
    return;
  }

  int source = 0;

  fscanf(fp, "%d", &no_of_nodes);

  // allocate host memory
  Node *h_graph_nodes = (Node *)malloc(sizeof(Node) * no_of_nodes);
  bool *h_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
  bool *h_updating_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
  bool *h_graph_visited = (bool *)malloc(sizeof(bool) * no_of_nodes);

  int start, edgeno;
  // initalize the memory
  for (unsigned int i = 0; i < no_of_nodes; i++) {
    fscanf(fp, "%d %d", &start, &edgeno);
    h_graph_nodes[i].starting = start;
    h_graph_nodes[i].no_of_edges = edgeno;
    h_graph_mask[i] = false;
    h_updating_graph_mask[i] = false;
    h_graph_visited[i] = false;
  }

  // read the source node from the file
  fscanf(fp, "%d", &source);
  // source=0; //tesing code line

  // set the source node as true in the mask
  h_graph_mask[source] = true;
  h_graph_visited[source] = true;

  fscanf(fp, "%d", &edge_list_size);

  int id, cost;
  int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
  for (int i = 0; i < edge_list_size; i++) {
    fscanf(fp, "%d", &id);
    fscanf(fp, "%d", &cost);
    h_graph_edges[i] = id;
  }

  if (fp)
    fclose(fp);

  // allocate mem for the result on host side
  int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
  for (int i = 0; i < no_of_nodes; i++)
    h_cost[i] = -1;
  h_cost[source] = 0;

  printf("Start traversing the tree\n");

#ifdef OPEN
  double start_time = omp_get_wtime();
#ifdef OMP_OFFLOAD
#pragma omp target data map(to : no_of_nodes, h_graph_mask[0 : no_of_nodes],   \
                                h_graph_nodes[0 : no_of_nodes],                \
                                h_graph_edges[0 : edge_list_size],             \
                                h_graph_visited[0 : no_of_nodes],              \
                                h_updating_graph_mask[0 : no_of_nodes])        \
    map(h_cost[0 : no_of_nodes])
  {
#endif
#endif
    __mc_profiling_begin();
    kernel_bfs(no_of_nodes, h_graph_mask, h_graph_nodes, h_graph_edges, h_cost,
               h_updating_graph_mask, h_graph_visited);
    __mc_profiling_end();
#ifdef OPEN
    double end_time = omp_get_wtime();
    printf("Compute time: %lf\n", (end_time - start_time));
#ifdef OMP_OFFLOAD
  }
#endif
#endif
  // Store the result into a file
  FILE *fpo = fopen("result.txt", "w");
  for (int i = 0; i < no_of_nodes; i++)
    fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
  fclose(fpo);
  printf("Result stored in result.txt\n");

  // cleanup memory
  free(h_graph_nodes);
  free(h_graph_edges);
  free(h_graph_mask);
  free(h_updating_graph_mask);
  free(h_graph_visited);
  free(h_cost);
}
