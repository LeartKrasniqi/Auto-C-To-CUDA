#include "rose.h"
#include <iostream>
#include "./include/loop_attr.hpp"
#include "./include/normalize/normalize.hpp"
#include "./include/affine/affine.hpp"
#include "./include/dependency/dependency.hpp"
#include "./include/parallel/parallel.hpp"
#define DEBUG 1


int main(int argc, char **argv)
{
	ROSE_INITIALIZE;
	SgProject *project = frontend(argc, argv);
	
	/* Get all function definitions */
	Rose_STL_Container<SgNode*> functions = NodeQuery::querySubTree(project, V_SgFunctionDefinition);
	Rose_STL_Container<SgNode*>::const_iterator func_iter = functions.begin();
	
	/* Will hold each of the loop nests */
	std::list<SgForStatement*> loop_nest_list;
	
	/* Loop through each function definition */
	while(func_iter != functions.end())
	{
		/* Get the actual definition node */
		SgFunctionDefinition* defn = isSgFunctionDefinition(*func_iter);

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
			
			/* TODO: Check if loop nest is perfectly nested */
			

			/* Obtain the attribute of the nest */
			LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));	
			

			/* Perform normalization */
			if(!normalizeLoopNest(loop_nest))
			{
				std::cout << "Loop Nest Skipped (Not Normalized)" << std::endl;
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
				std::cout << "Loop Nest Skipped (Not Affine)" << std::endl;
				attr->set_nest_flag(false);
				continue;
			}


			/* Dependency Tests */
			switch(dependencyExists(loop_nest))
			{
				case 0: /* TODO: Code Generation */
					std::cout << "No Dependency Exists" << std::endl;
					break;
				
				case 1: /* TODO: Parallelism Extraction */
					std::cout << "Dependency Exists" << std::endl;
					extractParallelism(loop_nest);

					break;
				
				case 2: /* Skip Loop Nest */
					std::cout << "Loop Nest Skipped (Could Not Determine Dependence)" << std::endl;
				        attr->set_nest_flag(false);
					continue;
				
				default: /* Should not reach here, skip loop to be safe */	
					continue;
			}

			
			/* TODO: Code Generation */
			
		}


		#if DEBUG
		std::cout << defn->unparseToString() << std::endl;	
		#endif
		

		func_iter++;
	}

	return 0;
}
