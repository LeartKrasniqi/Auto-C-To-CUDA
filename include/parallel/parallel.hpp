/* Header for parallelism extraction phase */

#ifndef PAR_EXTRACT
#define PAR_EXTRACT
#include <cmath>
#include <climits>
#include "rose.h"
#include "../loop_attr.hpp"
#include "../dependency/dependency.hpp"
#include "../kernel/kernel.hpp"

/* Class definition to represent dependency graph of a loop nest */
class Graph
{
	private:
		int num_vertices;	/* Number of vertices */
		std::list<int> *adj;	/* Array of adjacency lists */

	public:
		/* Constructor */
		Graph(int num_v) {this->num_vertices = num_v; this->adj = new std::list<int>[num_v];}
		

		/* Add edge to specific vertex */
		void addEdge(int vertex, int new_vertex) {adj[vertex].push_back(new_vertex);}


		/* Getter function to obtain the array of adjacency lists for the vertices in the graph */
		std::list<int> * getADJ() {return adj;}
		

		/* Getter function to obtain size of graph (i.e. number of vertices) */
		int getSize() {return num_vertices;}	


		/* Fills stack with vertices */
		void fillOrder(int vertex, bool *visited, std::stack<int> &stack)
		{
			/* Mark current vertex as visited */
			visited[vertex] = true;

			/* Go thru all vertices adjacent to this one and repeat */
			for(auto it = adj[vertex].begin(); it != adj[vertex].end(); it++)
				if(!visited[*it])
					fillOrder(*it, visited, stack);

			/* Once we get here, all the vertices reachable from vertex have been handled, so push vertex onto the stack */
			stack.push(vertex);
		}


		/* Obtain transpose of graph */
		Graph *getTranspose()
		{
			/* Create new graph to hold transpose */
			Graph *g = new Graph(num_vertices);

			/* Go thru each vertex and its adjacent vertices */
			for(int v = 0; v < num_vertices; v++)
				for(auto it = adj[v].begin(); it != adj[v].end(); it++)
					g->adj[*it].push_back(v);

			return g;
		}
		

		/* Recursive function that appends adjacent vertices to scc_list */
		void DFSUtil(int vertex, bool *visited, std::list<int> &scc_list)
		{
			/* Mark current vertex as visited and append to scc_list */
			visited[vertex] = true;
			scc_list.push_back(vertex);

			/* Go thru all vertices adjacent to this one and repeat */
			for(auto it = adj[vertex].begin(); it != adj[vertex].end(); it++)
				if(!visited[*it])
					DFSUtil(*it, visited, scc_list);
		}

		
		/* Retrieve list of strongly connected components (SCCs) using Kosaraju's Algorithm */
		std::list<std::list<int>> getSCCs()
		{
			/* List which will hold list of SCCs for each vertex */
			std::list<std::list<int>> scc_list;
			
			/* Stack which will be used to hold the vertices */
			std::stack<int> stack;

			/* Mark all vertices as not visited (for the first DFS) */
			bool *visited = new bool[num_vertices];
			for(int i = 0; i < num_vertices; i++)
				visited[i] = false;

			/* Perform first DFS to fill stack */
			for(int i = 0; i < num_vertices; i++)
				if(!visited[i])
					fillOrder(i, visited, stack);

			/* Now, get a transposed graph and mark vertices as not visited (for second DFS) */
			Graph *gt = getTranspose();
			for(int i = 0; i < num_vertices; i++)
				visited[i] = false;

			/* Process all vertices in the order of the stack */
			while(!stack.empty())
			{
				/* Obtain top vertex */
				int v = stack.top();
				stack.pop();

				/* Obtain SCC of popped vertex */
				if(!visited[v])
				{
					/* List to hold SCC for the popped vertex */
					std::list<int> scc_vertex_list;
					
					/* Perform the second DFS */
					gt->DFSUtil(v, visited, scc_vertex_list);
				
					/* Append the vertex SCC list to master list */
					scc_list.push_back(scc_vertex_list);
				}
			}

			return scc_list;
		}
	
};


/* Function to get dependency graph of loop nest body 

   Input: Body of loop nest
   Output: Dependency graph listing all flow/true dependencies 

*/
Graph * getDependencyGraph(SgBasicBlock *body);


/* Function that attempts to extract parallelism from a loop nest with dependencies 
   
   Input: Loop nest, global scope info, id of nest, flag for creation of ecsMinFn and ecsMaxFn
   Output: true if successful, false if not
 
   This function creates a dependency graph of the statements in the body of the loop nest.
   Using this graph, it attempts to extract any parallelism.
   It then makes calls to the kernel code generation function defined in ../kernel/kernel.cpp
*/
bool extractParallelism(SgForStatement *loop_nest, SgGlobal *globalScope, int &nest_id, bool &ecs_fn_flag);


/* Performs loop fission (i.e. splitting up loop into multiple loops) if dependencies allow for it
   
   Input: Loop nest, nest id, global scope info
   Output: true if successful, false if not

   This function will ultimately call a kernel code generation function.
*/
bool loopFission(SgForStatement *loop_nest, int &nest_id, SgGlobal *globalScope);



/* Perform extended cycle shrinking to extract parallelism from loop with dependencies 

   Input: Loop nest, list of SCCs, list of adj_lists, global_scope info, nest id info, ECS fn flag info
   Output: true if successful, false if not

   This function computes a distance vector for the loop and partitions the body into cycles, which can be executed in parallel.
   It also creates two functions, ecsMinFn() and ecsMaxFn() which are used as part of the algorithm, and inserts them into global scope (If they have not been created already).
   This function will ultimately call a kernel code generation function.
*/
bool extendedCycleShrink(SgForStatement *loop_nest, std::list<std::list<int>> scc_list, std::list<int> *adj_list, SgGlobal *globalScope, int &nest_id, bool &ecs_fn_flag);



/* Compute Dependence Distance Vector for two array references
   
   Input: write reference, read reference, ddv_arr (will be written to), loop nest attribute
   Output: true if successful, false if not

   This function supports references of the form:   a*i + a0, a*i + b0
   This function is used by extendedCycleShrink().
*/
bool computeDDV(SgPntrArrRefExp *w_arr_ref, SgPntrArrRefExp *r_arr_ref, std::vector<int> *ddv_arr, LoopNestAttribute *attr);

/* Helper functions to obtain useful info from numerical values during the ECS algorithm */
int getSign(int num);
double getAbs(double num);
int getCeil(double num);

/* Helper function to create the min/max function definitions required for the ECS algorithm */
void createECSFn(std::string name, SgGlobal *globalScope);

#endif
