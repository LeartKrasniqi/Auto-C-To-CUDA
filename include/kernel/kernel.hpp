/* Header file for CUDA kernel code generation */

#ifndef CUDA_KERNEL_GEN
#define CUDA_KERNEL_GEN
#include "rose.h"
#include "../loop_attr.hpp"



/* Driver function to create kernel definition and call for simple (i.e. NO DEPENDENCIES) cases and loop fission cases */ 
void kernelCodeGenSimple(SgForStatement *loop_nest, SgGlobal *globalScope, int &nest_id);

/* Driver function to create kernel definition and call for ECS cases */
SgStatement * kernelCodeGenECS(SgForStatement *serial_loop, std::vector<SgForStatement*> parallel_loops, std::vector<SgBasicBlock*> parallel_loop_bbs, int &nest_id, SgGlobal* globalScope);
//SgStatement * kernelCodeGenECS(SgForStatement *serial_loop, std::vector<SgForStatement*> parallel_loops, int &nest_id, SgGlobal* globalScope);

/* Function to create kernel definition in global scope */
void kernelFnDef(SgForStatement *loop_nest, std::vector<std::string> iter_vec, std::vector<SgExpression*> bound_vec, std::set<SgInitializedName*> param_vars, SgBasicBlock *body, int nest_id, SgGlobal *globalScope);

/* Function to make calls to cudaMalloc() and cudaMemcpy() */
SgExpression * kernelCUDAMalloc(SgBasicBlock *bb, SgInitializedName *arr_name, SgBasicBlock *parentScope);

/* Function to create kernel call */
std::vector<SgStatement*> kernelFnCall(SgForStatement *loop_nest, std::set<SgInitializedName*> param_vars, int nest_id);

/* Helper function to get relevant info (e.g. iter_vars, bounds_exprs, symb_vars, param_vars) for the loop nest */
// in simple case, loop_body = loop_nest, in ECS case, loop_body = parallel_loop_bbs[i]
bool getLoopInfo(SgForStatement *loop_nest, SgStatement *loop_body, std::vector<std::string> &iter_vec, std::vector<SgExpression*> &bound_vec, std::vector<SgInitializedName*> &symb_vec, std::set<SgInitializedName*> &param_vars);
//bool getLoopInfo(SgForStatement *loop_nest, std::vector<std::string> &iter_vec, std::vector<SgExpression*> &bound_vec, std::vector<SgInitializedName*> &symb_vec, std::set<SgInitializedName*> &param_vars);
//bool getParamVars(SgForStatement *loop_nest, std::set<SgInitializedName*> &param_vars);



#if 0
SgExpression * kernelCUDAMalloc(/*SgForStatement *loop_nest*/ SgBasicBlock *bb, SgInitializedName *arr_name, SgBasicBlock *parentScope);

/*SgBasicBlock * */ std::vector<SgStatement*> kernelFnCall(SgForStatement *loop_nest, std::set<SgInitializedName*> param_vars, int nest_id);

/* Function to create kernel definition */
//void kernelFnDef(SgForStatement *loop_nest, std::vector<std::string> iter_vec, std::vector<SgExpression*> bound_vec, std::vector<SgInitializedName*> symb_vec, std::set<SgInitializedName*> reads, std::set<SgInitializedName*> writes, SgBasicBlock *body, int nest_id, SgGlobal *globalScope);
#endif



#endif
