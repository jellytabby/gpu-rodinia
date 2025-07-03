#include "bfs_module.h"

void kernel_bfs(int no_of_nodes, bool *h_graph_mask, Node *h_graph_nodes,
                int *h_graph_edges, int *h_cost, bool *h_updating_graph_mask,
                bool *h_graph_visited) {
  int k = 0;
  bool stop;
  do {
    // if no thread changes this value then the loop stops
    stop = false;

#ifdef OPEN
    // omp_set_num_threads(num_omp_threads);
#ifdef OMP_OFFLOAD
#pragma omp target
#endif
#pragma omp parallel for
#endif
    for (int tid = 0; tid < no_of_nodes; tid++) {
      if (h_graph_mask[tid] == true) {
        h_graph_mask[tid] = false;
        for (int i = h_graph_nodes[tid].starting;
             i < (h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting);
             i++) {
          int id = h_graph_edges[i];
          if (!h_graph_visited[id]) {
            h_cost[id] = h_cost[tid] + 1;
            h_updating_graph_mask[id] = true;
          }
        }
      }
    }

#ifdef OPEN
#ifdef OMP_OFFLOAD
#pragma omp target map(stop)
#endif
#pragma omp parallel for
#endif
    for (int tid = 0; tid < no_of_nodes; tid++) {
      if (h_updating_graph_mask[tid] == true) {
        h_graph_mask[tid] = true;
        h_graph_visited[tid] = true;
        stop = true;
        h_updating_graph_mask[tid] = false;
      }
    }
    k++;
  } while (stop);
}
