// Structure to hold a node information
typedef struct Node {
  int starting;
  int no_of_edges;
} Node;

void kernel_bfs(int no_of_nodes, bool *h_graph_mask, Node *h_graph_nodes,
                int *h_graph_edges, int *h_cost, bool *h_updating_graph_mask,
                bool *h_graph_visited);
