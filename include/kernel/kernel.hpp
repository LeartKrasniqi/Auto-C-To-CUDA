/* Header file for CUDA kernel code generation */

#ifndef CUDA_KERNEL_GEN
#define CUDA_KERNEL_GEN
#include "rose.h"
#include "../loop_attr.hpp"



/* Driver function to create kernel definition and call */ 
void kernelCodeGenSimple(SgForStatement *loop_nest, SgGlobal *globalScope, int &nest_id); //, std::string ker_type);

SgExpression * kernelCUDAMalloc(/*SgForStatement *loop_nest*/ SgBasicBlock *bb, SgInitializedName *arr_name, SgBasicBlock *parentScope);

/*SgBasicBlock * */ std::vector<SgStatement*> kernelFnCall(SgForStatement *loop_nest, std::set<SgInitializedName*> param_vars, int nest_id);

/* Function to create kernel definition */
//void kernelFnDef(SgForStatement *loop_nest, std::vector<std::string> iter_vec, std::vector<SgExpression*> bound_vec, std::vector<SgInitializedName*> symb_vec, std::set<SgInitializedName*> reads, std::set<SgInitializedName*> writes, SgBasicBlock *body, int nest_id, SgGlobal *globalScope);

void kernelFnDef(SgForStatement *loop_nest, std::vector<std::string> iter_vec, std::vector<SgExpression*> bound_vec, std::set<SgInitializedName*> param_vars, SgBasicBlock *body, int nest_id, SgGlobal *globalScope);










#endif
