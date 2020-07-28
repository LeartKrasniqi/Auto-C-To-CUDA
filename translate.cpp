#include "rose.h"
#include <iostream>
#include "./include/loop_attr.hpp"
#include "./include/normalize/normalize.hpp"
#include "./include/affine/affine.hpp"
#include "./include/dependency/dependency.hpp"
#include "./include/parallel/parallel.hpp"
#include "./include/kernel/kernel.hpp"
#include "./include/preprocess/preprocess.hpp"
#define DEBUG 1


void printMsg(std::string msg)
{
	#if DEBUG
	std::cout << msg << std::endl;
	#endif
}

int main(int argc, char **argv)
{
	ROSE_INITIALIZE;
	SgProject *project = frontend(argc, argv);

	/* Obtain the global scope */
	SgGlobal *globalScope = SageInterface::getFirstGlobalScope(project);
	
	/* Get all function definitions */
	Rose_STL_Container<SgNode*> functions = NodeQuery::querySubTree(project, V_SgFunctionDefinition);
	Rose_STL_Container<SgNode*>::const_iterator func_iter = functions.begin();
	
	/* Will hold each of the loop nests */
	std::list<SgForStatement*> loop_nest_list;

	/* Will hold the id number of nests that will be parallelized (to be used to name kernel function) */
	int nest_id = 0;

	/* Flag to see if ecsMinFn and ecsMaxFn have been created already (to be used in parallelism extraction) */
	bool ecs_fn_flag = false;
	
	/* Loop through each function definition */
	while(func_iter != functions.end())
	{
		/* Get the actual definition node */
		SgFunctionDefinition* defn = isSgFunctionDefinition(*func_iter);

		/* Query for any while loops and try to convert them into for loops */
		Rose_STL_Container<SgNode*> whileLoops = NodeQuery::querySubTree(defn, V_SgWhileStmt);
		for(auto while_iter = whileLoops.begin(); while_iter != whileLoops.end(); /* EMPTY -- Increment at end of loop */)
		{
			/* Get the outer most while loop */
			SgWhileStmt *loop_nest = isSgWhileStmt(*while_iter);

			/* Find loop nest size */
			Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgWhileStmt);
			int nest_size = inner_loops.size();

			/* Perform the conversion */
			SgBasicBlock *for_loop_nest = isSgBasicBlock(convertWhileToFor(loop_nest));

			/* If successful, replace the loop nest with the for_loop */
			if(for_loop_nest)
				isSgStatement(loop_nest->get_parent())->replace_statement(loop_nest, for_loop_nest);

			/* Increment to get to next loop nest */
			while_iter += nest_size;
		}
		
		
		/* Query for the for loops */
		Rose_STL_Container<SgNode*> forLoops = NodeQuery::querySubTree(defn,V_SgForStatement);
		
		/* Obtain the loop nests */
		Rose_STL_Container<SgNode*>::const_iterator for_iter = forLoops.begin();
		while(for_iter != forLoops.end())
		{
			/* Get the outer most loop */
			SgForStatement* loop_nest = isSgForStatement(*for_iter);
			
			/* Find loop nest size */
			Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
			int nest_size = inner_loops.size();
			
			/* Set attributes (applied to outermost loop) */
			loop_nest->setAttribute("LoopNestInfo", new LoopNestAttribute(nest_size, true));
			
			/* Append loop_nest to list of loop nests */
			loop_nest_list.push_back(loop_nest);
			
			/* Increment to get to next loop_nest */
			for_iter += nest_size;
		}

			
		/* Iterate through the loop nests */
		std::list<SgForStatement*>::iterator nest_iter; 
		for(nest_iter = loop_nest_list.begin(); nest_iter != loop_nest_list.end(); nest_iter++)
		{	
			SgForStatement *loop_nest = *nest_iter;
			
			/* Check if loop nest is perfectly nested */
			Rose_STL_Container<SgNode*> total_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
			bool imperf = false;
			for(size_t idx = 0; idx < total_loops.size() - 1; idx++)
			{
				SgStatementPtrList &loop_stmts = isSgBasicBlock(isSgForStatement(total_loops[idx])->get_loop_body())->get_statements();
				if(loop_stmts.size() > 1)
				{
					printMsg("Imperfect Loop");
					imperf = true;
					break;
				}
			}

			if(imperf)
			{
				//convertImperfToPerf(loop_nest)
				continue;
			}


			/* Obtain the attribute of the nest */
			LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));	


			/* Perform normalization */
			if(!normalizeLoopNest(loop_nest))
			{
				//std::cout << "Loop Nest Skipped (Not Normalized)" << std::endl;
				printMsg("Loop Nest Skipped (Not Normalized)");
				attr->set_nest_flag(false);
				continue;
			}
			
			/* Obtain the iteration, bound, and symbolic_constant vectors for the loop nest */
			std::list<std::string> iter_vec, symb_vec;
			std::list<SgExpression*> bound_vec;
			Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
			Rose_STL_Container<SgNode*>::iterator inner_it;
			for(inner_it = inner_loops.begin(); inner_it != inner_loops.end(); inner_it++)
			{
				SgForStatement *l = isSgForStatement(*inner_it);
				
				/* Iteration variables */
				iter_vec.push_back( SageInterface::getLoopIndexVariable(l)->get_name().getString() );
				
				/* Bounds Expressions */
				SgExpression *bound = isSgBinaryOp(l->get_test_expr())->get_rhs_operand();
				bound_vec.push_back(bound);

				/* Symbolic Constants -- Query for any variable references in the bounds expression */
				Rose_STL_Container<SgNode*> v = NodeQuery::querySubTree(bound, V_SgVarRefExp);
				for(Rose_STL_Container<SgNode*>::iterator v_it = v.begin(); v_it != v.end(); v_it++)
				{
					std::string var_name = isSgVarRefExp(*v_it)->get_symbol()->get_name().getString();
					
					/* Keep only unique vars */
					if( std::find(symb_vec.begin(), symb_vec.end(), var_name) != symb_vec.end() )
						continue;
					else
						symb_vec.push_back(var_name);
					
				}

				
			}

				
			/* Append iter_vec, bound_vec, and symb_vec to attributes */
			attr->set_iter_vec(iter_vec);
			attr->set_bound_vec(bound_vec);
			attr->set_symb_vec(symb_vec);	
			
			/* Affine test */
			if(!affineTest(loop_nest))
			{
				printMsg("Loop Nest Skipped (Not Affine)");
				attr->set_nest_flag(false);
				continue;
			}


			/* Dependency Tests */
			switch(dependencyExists(loop_nest))
			{
				case 0: /* Code Generation */
					printMsg("No Dependency Exists");
					kernelCodeGenSimple(loop_nest, globalScope, nest_id);
					break;
				
				case 1: /* Parallelism Extraction */
					printMsg("Dependency Exists");
					if(!extractParallelism(loop_nest, globalScope, nest_id, ecs_fn_flag))
						printMsg("Loop Nest Skipped (Could Not Extract Parallelism");

					break;
				
				case 2: /* Skip Loop Nest */
					printMsg("Loop Nest Skipped (Could Not Determine Dependence)");
				        attr->set_nest_flag(false);
					continue;
				
				default: /* Should not reach here, skip loop to be safe */	
					continue;
			}

			
			
			
		}


		#if DEBUG
		std::cout << defn->unparseToString() << std::endl;	
		#endif
		

		func_iter++;
	}
	
	/* Get all newly created kernel function definitions and add __global__ or __device__ in front of them */
	/*
	Rose_STL_Container<SgNode*> kernel_fns = NodeQuery::querySubTree(project, V_SgFunctionDeclaration);
	for(auto k_it = kernel_fns.begin(); k_it != kernel_fns.end(); k_it++)
	{
		SgFunctionDeclaration *k_fn = isSgFunctionDeclaration(*k_it);
		SgFunctionModifier &k_mod = k_fn->get_functionModifier();
		if(k_mod.isCudaGlobalFunction())
			SageInterface::addTextForUnparser(k_fn, "__global__ ", AstUnparseAttribute::e_before);
		else if(k_mod.isCudaDevice())
			SageInterface::addTextForUnparser(k_fn, "__device__ ", AstUnparseAttribute::e_before);
	}*/

	/* #define the CUDA_BLOCKs */
	SageBuilder::buildCpreprocessorDefineDeclaration(globalScope, "#define CUDA_BLOCK_X 128");
	SageBuilder::buildCpreprocessorDefineDeclaration(globalScope, "#define CUDA_BLOCK_Y 1");
	SageBuilder::buildCpreprocessorDefineDeclaration(globalScope, "#define CUDA_BLOCK_Z 1");

	/* Obtain translation */
	project->unparse();

	return 0;
}
