/* Implementation of building kernel functions/calls */
#include "./kernel.hpp"


/* Driver function to create kernel defn/call */
void kernelCodeGen(SgForStatement *loop_nest, SgGlobal *globalScope, int &nest_id)
{
	/* Build the basic block which will replace the loop nest */
	SgBasicBlock *bb_new = SageBuilder::buildBasicBlock();
	
	/* Obtain the iter_vars and bound_exprs for the loop nest -- Cannot just use loop attr because we have made transformations */
	std::vector<std::string> iter_vec;
	std::vector<SgExpression*> bound_vec;
	std::vector<SgInitializedName*> symb_vec;
	Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
	for(auto inner_it = inner_loops.begin(); inner_it != inner_loops.end(); inner_it++)
	{
		SgForStatement *l = isSgForStatement(*inner_it);

		/* Iteration variables */
		iter_vec.push_back( SageInterface::getLoopIndexVariable(l)->get_name().getString() );
														
		/* Bounds Expressions */
		SgExpression *bound = isSgBinaryOp(l->get_test_expr())->get_rhs_operand();
		bound_vec.push_back(bound);

		/* Symbolic variables */
		Rose_STL_Container<SgNode*> v = NodeQuery::querySubTree(bound, V_SgVarRefExp);
		for(auto v_it = v.begin(); v_it != v.end(); v_it++)
		{
			SgInitializedName *var_name = isSgVarRefExp(*v_it)->get_symbol()->get_declaration();
																										
			/* Keep only unique vars */
			if( std::find(symb_vec.begin(), symb_vec.end(), var_name) != symb_vec.end() )
				continue;
			else	
				symb_vec.push_back(var_name);
		}
	}

	/* Obtain loop_nest body -- A copy of this will be used in the body of the kernel function */
	SgBasicBlock *body = isSgBasicBlock(isSgForStatement(inner_loops[inner_loops.size() - 1])->get_loop_body());
	SgBasicBlock *kernel_body = isSgBasicBlock(SageInterface::copyStatement(body));

	/* Find read/write vars so that we copy the proper variables to the GPU */
	std::set<SgInitializedName*> reads, writes;
	SageInterface::collectReadWriteVariables(loop_nest, reads, writes);
	
	/* Create a new set that only includes relevant variables and removes duplicates */
	std::set<SgInitializedName*> param_vars;

	for(auto r_it = reads.begin(); r_it != reads.end(); r_it++)
		if( std::find(iter_vec.begin(), iter_vec.end(), (*r_it)->get_name().getString()) == iter_vec.end() )
			param_vars.insert(*r_it);

	for(auto w_it = writes.begin(); w_it != writes.end(); w_it++)
		if( std::find(iter_vec.begin(), iter_vec.end(), (*w_it)->get_name().getString()) == iter_vec.end() )
			param_vars.insert(*w_it);

	for(auto s_it = symb_vec.begin(); s_it != symb_vec.end(); s_it++)
		param_vars.insert(*s_it);

	
	/* Define kernel function */
	//kernelFnDef(loop_nest, iter_vec, bound_vec, symb_vec, reads, writes, kernel_body, nest_id, globalScope);
	kernelFnDef(loop_nest, iter_vec, bound_vec, param_vars, kernel_body, nest_id, globalScope);
	nest_id += 1;

	/* Create kernel call -- Probably replace loop_nest with this statement */
	//kernelFnCall()
	




}

/* Create the kernel function definition at the top of global scope */
void kernelFnDef(SgForStatement *loop_nest, std::vector<std::string> iter_vec, std::vector<SgExpression*> bound_vec, std::set<SgInitializedName*> param_vars, SgBasicBlock *body, int nest_id, SgGlobal *globalScope)
{
	std::string kernel_name = "_auto_kernel_" + std::to_string(nest_id);
	
	/* Create the parameter list and copy the relevant variables (reads, writes, symb_vec) */
	SgFunctionParameterList *kernel_params = SageBuilder::buildFunctionParameterList();
	
	/* Need to set them as new variables */ 
	for(auto p_it = param_vars.begin(); p_it != param_vars.end(); p_it++)
	{
		SgInitializedName *param = SageBuilder::buildInitializedName( (*p_it)->get_name(), (*p_it)->get_type() );
		SageInterface::appendArg(kernel_params, param);
	}
	
#if 0	
	for(auto r_it = reads.begin(); r_it != reads.end(); r_it++)
		SageInterface::appendArg(kernel_params, *r_it);
	
	for(auto w_it = writes.begin(); w_it != writes.end(); w_it++)
		SageInterface::appendArg(kernel_params, *w_it);

	for(auto symb_it = symb_vec.begin(); symb_it != symb_vec.begin(); symb_it++)
		SageInterface::appendArg(kernel_params, *symb_it);
#endif	

	/* Create the function declaration and apply function modifier to make it a global kernel function */
	SgFunctionDeclaration *kernel_fn = SageBuilder::buildDefiningFunctionDeclaration(kernel_name, SageBuilder::buildVoidType(), kernel_params, globalScope);
	SgFunctionModifier &kernel_mod = kernel_fn->get_functionModifier();
	kernel_mod.setCudaGlobalFunction();

	/* Add statements to body of kernel function */
	SgBasicBlock *kernel_body = kernel_fn->get_definition()->get_body();
	
	/* First, get the thread_ids */
	switch(iter_vec.size())
	{
		default:	/* > 3, fall through */

		case 3:
		{		
			/* Make z thread and fall through */	
			SgVariableDeclaration *thread_z = SageBuilder::buildVariableDeclaration("thread_z_id", SageBuilder::buildIntType(), NULL, kernel_body);
			SageInterface::addTextForUnparser(thread_z, "thread_z_id = blockIdx.z * blockDim.z + threadIdx.z;\n", AstUnparseAttribute::e_after);
			SageInterface::prependStatement(thread_z, kernel_body);
		}

		case 2:	
		{	
			/* Make y thread and fall through */
			SgVariableDeclaration *thread_y = SageBuilder::buildVariableDeclaration("thread_y_id", SageBuilder::buildIntType(), NULL, kernel_body);
			SageInterface::addTextForUnparser(thread_y, "thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;\n", AstUnparseAttribute::e_after);
			SageInterface::prependStatement(thread_y, kernel_body);
		}

		case 1:
		{
			/* Make x thread and break */
			SgVariableDeclaration *thread_x = SageBuilder::buildVariableDeclaration("thread_x_id", SageBuilder::buildIntType(), NULL, kernel_body);
			SageInterface::addTextForUnparser(thread_x, "thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;\n", AstUnparseAttribute::e_after);
			SageInterface::prependStatement(thread_x, kernel_body);
			break;
		}

	}

	/* Create a vector holding the var refs to the threads created -- [0] = x, [1] = y, [2] = z */
	std::vector<SgVarRefExp*> thread_refs;
	thread_refs.push_back(SageBuilder::buildVarRefExp("thread_x_id", kernel_body));
	if(iter_vec.size() >= 2)
	{
		thread_refs.push_back(SageBuilder::buildVarRefExp("thread_y_id", kernel_body));
		if(iter_vec.size() >= 3)
			thread_refs.push_back(SageBuilder::buildVarRefExp("thread_z_id", kernel_body));
	}
	

	/* Since our loops have a LB of 1, we need to make sure the thread_ids are not zero */
	SgExpression *non_zero_test, *bound_test;
	if(thread_refs.size() == 1)
	{
		non_zero_test = thread_refs[0];
		bound_test = SageBuilder::buildLessOrEqualOp(thread_refs[0], bound_vec[0]);
	}
	else if(thread_refs.size() == 2)
	{
		non_zero_test = SageBuilder::buildAndOp(thread_refs[0], thread_refs[1]);
		bound_test = SageBuilder::buildAndOp(
						SageBuilder::buildLessOrEqualOp(thread_refs[0], bound_vec[0]),
						SageBuilder::buildLessOrEqualOp(thread_refs[1], bound_vec[1]) );
	}
	else
	{
		non_zero_test = SageBuilder::buildAndOp(thread_refs[0], SageBuilder::buildAndOp(thread_refs[1], thread_refs[2]));
		bound_test = SageBuilder::buildAndOp(
						SageBuilder::buildLessOrEqualOp(thread_refs[0], bound_vec[0]),
						SageBuilder::buildAndOp(
								SageBuilder::buildLessOrEqualOp(thread_refs[1], bound_vec[1]),
								SageBuilder::buildLessOrEqualOp(thread_refs[2], bound_vec[2]) ) );
	}

	/* Build the bound_if -- The true body is the body of the loop nest, with appropriate addition of loops */
	SgBasicBlock *bound_if_body = body;
	if(iter_vec.size() > 3)
	{
		Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
		bound_if_body = isSgBasicBlock(isSgForStatement(inner_loops[2])->get_loop_body());
		
		// TODO: Create declarations for all of the iter vars, if they do not exist already
	}
	
	/* Change references to iter vars within the bound_if_body */
	Rose_STL_Container<SgNode*> iter_var_refs = NodeQuery::querySubTree(bound_if_body, V_SgVarRefExp);
	for(auto ref_it = iter_var_refs.begin(); ref_it != iter_var_refs.end(); ref_it++)
	{
		std::string ref_name = isSgVarRefExp(*ref_it)->get_symbol()->get_name().getString();
		
		/* Find if the var ref is an iter_var */
		auto iter_it = std::find(iter_vec.begin(), iter_vec.end(), ref_name); 
		if(iter_it != iter_vec.end())
		{
			/* If so, find index of the iter var */
			int iter_index = std::distance(iter_vec.begin(), iter_it);
			
			/* If the index is <= 2, then we have made that index a thread, so replace the proper expression */
			if(iter_index <= 2)
				SageInterface::replaceExpression(isSgVarRefExp(*ref_it), thread_refs[iter_index]);

		}
	}

	/* Build the if */
	SgIfStmt *bound_if = SageBuilder::buildIfStmt(bound_test, bound_if_body, NULL);

	/* Set the bound_if as the body of the non_zero_if */
	SgIfStmt *non_zero_if = SageBuilder::buildIfStmt(non_zero_test, bound_if, NULL);

	/* Append non_zero_if to the body of the function */
	SageInterface::appendStatement(non_zero_if, kernel_body);
	
	/* Fix any var refs in kernel_body */
	SageInterface::fixVariableReferences(kernel_body);
	
	/* Prepend function to global scope */
	SageInterface::prependStatement(kernel_fn, globalScope);
	
}

void kernelFnCal()
{
	//buildCudaKernelExecConfig_nfi()
	//buildCudaKernelCallExp_nfi()

}

/* SINCE WE START INDEX AT 1, NEED TO ADD CHECK: if(thread_id_1 != 0 && thread_id_2 != 0 && thread_id_3 != 0) */
