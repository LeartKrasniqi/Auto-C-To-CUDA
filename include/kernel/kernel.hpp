/* Header file for CUDA kernel code generation */

#ifndef CUDA_KERNEL_GEN
#define CUDA_KERNEL_GEN
#include "rose.h"
#include "../loop_attr.hpp"



/* Driver function to create kernel definition and call for simple (i.e. NO DEPENDENCIES) cases and loop fission cases
   
   Input: Loop nest, global scope info, nest_id (passed by reference so that we can update it throughout the code base as needed)
   Output: Generates kernel fn defn, calls to cudaMalloc() and cudaMemcpy(), and calls to the kernel itself

   This function is used for the NO DEPENDENCIES and loop fission cases.  For the ECS cases, a different driver function is used.
*/ 
void kernelCodeGenSimple(SgForStatement *loop_nest, SgGlobal *globalScope, int &nest_id);


/* Driver function to create kernel definition and call for ECS cases 
 
   Input: Outer-most serial loop, vector of inner-most parallel loops, vector of basic blocks containing the statements within the parallel loops, nest id, global scope info
   Output: SgStatement* representing the transformed loop

   The SgStatement* that is returned is ultimately used in ../parallel/parallel.cpp in extendedCycleShrink().
   Each SCC which undergoes the ECS algorithm produces a SgStatement* containing the outer-most serial loop and then a series of kernel calls.
   In the end, the loop nest is replaced by a series of these SgStatement*'s.
*/
SgStatement * kernelCodeGenECS(SgForStatement *serial_loop, std::vector<SgForStatement*> parallel_loops, std::vector<SgBasicBlock*> parallel_loop_bbs, int &nest_id, SgGlobal* globalScope);


/* Function to create kernel definition in global scope 

   Input: loop nest, iter_vec, bound_vec, parameter variables (i.e. those that will be cudaMalloc()'ed and cudaMemcpy()'ed), body of loop that will become body of function, nest id, global scope info
   Output: CUDA kernel function is created in global scope

   This function uses nest_id to determine the name of the kernel function: _auto_kernel_{nest_id} 
*/
void kernelFnDef(SgForStatement *loop_nest, std::vector<std::string> iter_vec, std::vector<SgExpression*> bound_vec, std::set<SgInitializedName*> param_vars, SgBasicBlock *body, int nest_id, SgGlobal *globalScope);


/* Function to make calls to cudaMalloc() and cudaMemcpy() 
 
   Input: basic block where the calls to cudaMalloc() and cudaMemcpy() will be made, name of array to be allocated space and copied, parent scope info
   Output: SgExpression* detailing the number of bytes that were allocated for the array (to be used in later calls to cudaMemcpy() when copying from device to host)

   This function only takes care of the initial cudaMemcpy() (i.e. host to device).
   The other cudaMemcpy() is done in the kernelCodeGen() driver functions
*/
SgExpression * kernelCUDAMalloc(SgBasicBlock *bb, SgInitializedName *arr_name, SgBasicBlock *parentScope);


/* Function to create kernel call 

   Input: loop nest, parameter variables, nest id
   Output: vector of statements holding the kernel call and relevant cudaMemcpy()'s

   This function calculates the CUDA_GRID sizes, depending on each of the array dimensions.
   Additionally, this function uses nest_id to call the proper kernel.
*/
std::vector<SgStatement*> kernelFnCall(SgForStatement *loop_nest, std::set<SgInitializedName*> param_vars, int nest_id);


/* Helper function to get relevant info (e.g. iter_vars, bounds_exprs, symb_vars, param_vars) for the loop nest 

   Input: loop nest, loop body, iter_vec, bound_vec, symb_vec, param_vars
   Output: True if there is at least one write to an array (i.e. it makes sense to parallelize the loop), false otherwise

   In the simple kernel gen case, loop_body = loop_nest.  In the ECS case, loop_body = parallel_loop_bbs[i].
   iter_vec, bound_vec, symb_vec, and param_vars are all passed by reference so that we can extract the relevant info with one function call. 
*/
bool getLoopInfo(SgForStatement *loop_nest, SgStatement *loop_body, std::vector<std::string> &iter_vec, std::vector<SgExpression*> &bound_vec, std::vector<SgInitializedName*> &symb_vec, std::set<SgInitializedName*> &param_vars);


#endif
