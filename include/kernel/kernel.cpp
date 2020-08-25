/* Implementation of building kernel functions/calls */
#include "./kernel.hpp"


/* Driver function to create kernel defn/call for simple code generation (i.e. NO DEPENDENCIES) */
void kernelCodeGenSimple(SgForStatement *loop_nest, SgGlobal *globalScope, int &nest_id)
{
	/* Build the basic block which will replace the loop nest */
	SgBasicBlock *bb_new = SageBuilder::buildBasicBlock();
	bb_new->set_parent(loop_nest->get_parent());

	/* Obtain relevant info from the loop nest */
	std::vector<std::string> iter_vec;
	std::vector<SgExpression*> bound_vec;
	std::vector<SgInitializedName*> symb_vec;
	std::set<SgInitializedName*> param_vars;
	if(!getLoopInfo(loop_nest, loop_nest, iter_vec, bound_vec, symb_vec, param_vars))
		return;

	/* Obtain loop_nest body -- A copy of this will be used in the body of the kernel function */
	Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
	SgBasicBlock *body = isSgBasicBlock(isSgForStatement(inner_loops[inner_loops.size() - 1])->get_loop_body());
	SgBasicBlock *kernel_body = isSgBasicBlock(SageInterface::copyStatement(body));

	/* Define kernel function */
	kernelFnDef(loop_nest, iter_vec, bound_vec, param_vars, kernel_body, nest_id, globalScope);
	
	/* Make calls to cudaMalloc() and cudaMemcpy() for each array */
	std::map<std::string, SgExpression*> bytes_map;
	for(auto p_it = param_vars.begin(); p_it != param_vars.end(); p_it++)
	{
		if(!isSgArrayType((*p_it)->get_type()))
			continue;

		SgExpression *num_bytes = kernelCUDAMalloc(bb_new, *p_it, isSgBasicBlock(loop_nest->get_parent()));
		
		bytes_map[(*p_it)->get_name().getString()] = num_bytes;
	}
	
	/* Create kernel call -- Probably replace loop_nest with this statement */
	//SgBasicBlock *kernel_call = kernelFnCall(loop_nest, param_vars, nest_id);
	std::vector<SgStatement*> kernel_stmts = kernelFnCall(loop_nest, param_vars, nest_id);
	
	/* Do the cudaMemcpy() from device to host */
	for(auto p_it = param_vars.begin(); p_it != param_vars.end(); p_it++)
	{
		if(!isSgArrayType((*p_it)->get_type()))
			continue;

		std::string arr_name = (*p_it)->get_name().getString(); 
		std::string d_arr_ptr_name = "d_" + arr_name;
		std::string memcpy_string = "\n    cudaMemcpy(" + arr_name + ", " + d_arr_ptr_name + ", " + bytes_map[arr_name]->unparseToString() + ", cudaMemcpyDeviceToHost);\n";
		SageInterface::addTextForUnparser(kernel_stmts.back(), memcpy_string, AstUnparseAttribute::e_after);	
	}
	
	/* Replace the loop nest with the kernel call */
	//isSgStatement(loop_nest->get_parent())->replace_statement(loop_nest, kernel_call);
	
	/* Add comment above first statement, to let user know that the code in this bb was auto-generated */
	std::string comment = "Auto-generated code for call to _auto_kernel_" + std::to_string(nest_id);
	SageInterface::attachComment(SageInterface::getFirstStatement(bb_new), comment);	

	/* Replace loop nest with bb_new, which contains the typedefs, mallocs, memcpys, and the kernel call */
	SageInterface::appendStatementList(kernel_stmts, bb_new);
	isSgStatement(loop_nest->get_parent())->replace_statement(loop_nest, bb_new);

	/* Don't forget to update the nest_id */
	nest_id += 1;
}

/* Driver function to generate CUDA for ECS cases */
SgStatement * kernelCodeGenECS(SgForStatement *serial_loop, std::vector<SgForStatement*> parallel_loops, std::vector<SgBasicBlock*> parallel_loop_bbs, int &nest_id, SgGlobal* globalScope)
{
	/* Map which will hold the bytes expression for the cudaMalloc() for each array */	
	std::map<SgInitializedName*, SgExpression*> bytes_map;

	/* Basic block which will hold the cudaMalloc()s, cudaMemcpy()s, serial_loop, parallel_loops, and final cudaMemcpy()s */
	SgBasicBlock *bb_main = SageBuilder::buildBasicBlock();
	bb_main->set_parent(serial_loop->get_parent());

	/* Save the initial nest id, for the comment which will go above the bb when the code is generated */
	int init_nest_id = nest_id;
	for(auto pl_it = parallel_loops.begin(); pl_it != parallel_loops.end(); pl_it++)
	{
		SgForStatement *curr_loop_nest = *pl_it;
		int bb_idx = std::distance(parallel_loops.begin(), pl_it);	
			
		/* Get relevant info for the loop nest -- Handle this a bit differently than simple case, due to weird thing with findReadWriteVariables() */
		std::vector<std::string> iter_vec;
		std::vector<SgExpression*> bound_vec;
		std::vector<SgInitializedName*> symb_vec;	
		std::set<SgInitializedName*> param_vars;
		if(!getLoopInfo(curr_loop_nest, parallel_loop_bbs[bb_idx], iter_vec, bound_vec, symb_vec, param_vars)) // pass parallel_loop_bbs[i] as loop_body
			return serial_loop;
			
	
		/* Obtain loop_nest body -- A copy of this will be used in the body of the kernel function */
		Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(curr_loop_nest, V_SgForStatement);
		SgBasicBlock *body = isSgBasicBlock(isSgForStatement(inner_loops[inner_loops.size() - 1])->get_loop_body());
		SgBasicBlock *kernel_body = isSgBasicBlock(SageInterface::copyStatement(body));
		
		/* Create the kernel definition */
		kernelFnDef(curr_loop_nest, iter_vec, bound_vec, param_vars, kernel_body, nest_id, globalScope);
		
		/* Perform the cudaMalloc() and cudaMemcpy() only ONCE (for each array), since all the statements refer to the same arrays */
		int index = std::distance(parallel_loops.begin(), pl_it);
		if(index == 0)
		{
			for(auto p_it = param_vars.begin(); p_it != param_vars.end(); p_it++)
			{
				if(!isSgArrayType((*p_it)->get_type()))
					continue;

				SgExpression *num_bytes = kernelCUDAMalloc(bb_main, *p_it, isSgBasicBlock(serial_loop->get_parent()));
		
				bytes_map[(*p_it)] = num_bytes;
			}
		}

		/* Make the kernel call */
		std::vector<SgStatement*> kernel_stmts = kernelFnCall(curr_loop_nest, param_vars, nest_id);
		
		/* Replace curr_loop_nest with the kernel_stmts */
		SgBasicBlock *bb_inner = SageBuilder::buildBasicBlock();
		bb_inner->set_parent(bb_main);
		SageInterface::appendStatementList(kernel_stmts, bb_inner);
		isSgStatement(curr_loop_nest->get_parent())->replace_statement(curr_loop_nest, bb_inner);
		
		/* Increment nest_id */
		nest_id += 1;
	}
	
	/* Append the serial loop to bb_main */
	SageInterface::appendStatement(serial_loop, bb_main);
	
	/* Perform the d2h cudaMemcpy()s after the serial loop */
	for(auto p_it = bytes_map.begin(); p_it != bytes_map.end(); p_it++)
	{
		if(!isSgArrayType(p_it->first->get_type()))
			continue;

		SgInitializedName *arr_init_name = p_it->first;
		std::string arr_name = arr_init_name->get_name().getString(); 
		std::string d_arr_ptr_name = "d_" + arr_name;
		std::string memcpy_string = "\n    cudaMemcpy(" + arr_name + ", " + d_arr_ptr_name + ", " + p_it->second->unparseToString() + ", cudaMemcpyDeviceToHost);\n";
		SageInterface::addTextForUnparser(serial_loop, memcpy_string, AstUnparseAttribute::e_after);	
	}
	
	/* Add comment above first statement, to let user know that the code in this bb was auto-generated */
	std::string comment = "Auto-generated code for calls to _auto_kernel_" + std::to_string(init_nest_id) + " to _auto_kernel_" + std::to_string(nest_id - 1);
	SageInterface::attachComment(SageInterface::getFirstStatement(bb_main), comment);	

	return bb_main;
}

/* Function to obtain parameter variables */
bool getLoopInfo(SgForStatement *loop_nest, SgStatement *loop_body, std::vector<std::string> &iter_vec, std::vector<SgExpression*> &bound_vec, std::vector<SgInitializedName*> &symb_vec, std::set<SgInitializedName*> &param_vars)
{
	/* Obtain the iter_vars and bound_exprs for the loop nest -- Cannot just use loop attr because we have made transformations */
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
		
	/* Find read/write vars so that we copy the proper variables to the GPU */
	std::set<SgInitializedName*> reads, writes;
	//SageInterface::collectReadWriteVariables(loop_nest, reads, writes);
	SageInterface::collectReadWriteVariables(loop_body, reads, writes);
	
	/* If there are no arrays in the writes, skip the loop */
	bool arr_writes = false;
	for(auto w_it = writes.begin(); w_it != writes.end(); w_it++)
		if( isSgArrayType((*w_it)->get_type()) )
			arr_writes = true;
	if(!arr_writes)
		return false;

	/* Get scope of loop body -- Used to see whether we need to pass a variable as a parameter to the kernel, or if it will be defined there itself */
	SgScopeStatement *inner_scope = isSgScopeStatement(loop_body);
	if(isSgForStatement(loop_body))
	{
		Rose_STL_Container<SgNode*> temp_inner_loops = NodeQuery::querySubTree(loop_body, V_SgForStatement);
		inner_scope = isSgScopeStatement(isSgForStatement(temp_inner_loops.back())->get_loop_body());
	}

	/* Append to the set that will only include relevant variables and removes duplicates */
	for(auto r_it = reads.begin(); r_it != reads.end(); r_it++)
		if( std::find(iter_vec.begin(), iter_vec.end(), (*r_it)->get_name().getString()) == iter_vec.end() )
			if( (*r_it)->get_scope() != inner_scope )
				param_vars.insert(*r_it);

	for(auto w_it = writes.begin(); w_it != writes.end(); w_it++)
		if( std::find(iter_vec.begin(), iter_vec.end(), (*w_it)->get_name().getString()) == iter_vec.end() )
			if( (*w_it)->get_scope() != inner_scope )
				param_vars.insert(*w_it);

	for(auto s_it = symb_vec.begin(); s_it != symb_vec.end(); s_it++)
		param_vars.insert(*s_it);

	return true;
}

/* Make the call to the kernel */
std::vector<SgStatement*> kernelFnCall(SgForStatement *loop_nest, std::set<SgInitializedName*> param_vars, int nest_id)
{
	std::vector<SgStatement*> stmt_list;

	/* Find the gridSize and blockSize values for each dimension (i.e. x,y,z) */
	int data_sizes[3] = {1,1,1};
	for(auto p_it = param_vars.begin(); p_it != param_vars.end(); p_it++)
	{
		/* Skip non-array parameters */
		SgArrayType *arr_type = isSgArrayType((*p_it)->get_type());
		if(!arr_type)
			continue;

		/* Obtain the dimension info for the array -- First expression is at index 1 */
		std::vector<SgExpression*> arr_dim_list = SageInterface::get_C_array_dimensions(arr_type, **p_it);

		/* If the dimension is an INT, then find the biggest x_dim, y_dim, and z_dim to be used to calculate the gridSize */
		for(size_t i = 1; (i < arr_dim_list.size()) && (i < 4); i++)
		{
			SgIntVal *dim_size = isSgIntVal(arr_dim_list[i]);
			if(dim_size)
				if(dim_size->get_value() > data_sizes[i-1])
					data_sizes[i-1] = dim_size->get_value();
		}
	}

	/* Create variables to represent the data sizes */
	SgVariableDeclaration *x_grid = SageBuilder::buildVariableDeclaration("CUDA_GRID_X", SageBuilder::buildIntType(), NULL, loop_nest);
	SageInterface::addTextForUnparser(x_grid, "\n    CUDA_GRID_X = (" + std::to_string(data_sizes[0]) + " + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;\n", AstUnparseAttribute::e_after); 
	
	SgVariableDeclaration *y_grid = SageBuilder::buildVariableDeclaration("CUDA_GRID_Y", SageBuilder::buildIntType(), NULL, loop_nest);
	SageInterface::addTextForUnparser(y_grid, "\n    CUDA_GRID_Y = (" + std::to_string(data_sizes[1]) + " + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;\n", AstUnparseAttribute::e_after); 

	SgVariableDeclaration *z_grid = SageBuilder::buildVariableDeclaration("CUDA_GRID_Z", SageBuilder::buildIntType(), NULL, loop_nest);
	SageInterface::addTextForUnparser(z_grid, "\n    CUDA_GRID_Z = (" + std::to_string(data_sizes[2]) + " + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;\n", AstUnparseAttribute::e_after); 
	
	stmt_list.push_back(x_grid);
	stmt_list.push_back(y_grid);
	stmt_list.push_back(z_grid);

	/* Create the blockSize and gridSize declarations */
	std::string block_decl = "    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);\n";
	std::string grid_decl = "    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);\n";

	/* Make the kernel function call */
	std::string kernel_call_string = "    _auto_kernel_" + std::to_string(nest_id) + "<<<CUDA_gridSize,CUDA_blockSize>>>(";
	for(auto p_it = param_vars.begin(); p_it != param_vars.end(); p_it++)
	{
		/* Differentiate between arrays and non-arrays */
		if(!isSgArrayType((*p_it)->get_type()))
			kernel_call_string += (*p_it)->unparseToString();
		else
			kernel_call_string += "d_" + (*p_it)->unparseToString();
		
		/* Add proper ending, depending on whether or not we have added all the param vars */
		if(std::next(p_it) == param_vars.end())
			kernel_call_string += ");\n";
		else
			kernel_call_string += ", ";
	}
	
	/* Add the block, grid, and kernel call */
	SageInterface::addTextForUnparser(z_grid, block_decl + grid_decl + kernel_call_string, AstUnparseAttribute::e_after);

	return stmt_list;
}


/* Allocate space for array variables on the GPU and copy them over */
SgExpression * kernelCUDAMalloc(SgBasicBlock *bb_new, SgInitializedName *arr_name, SgBasicBlock *parentScope)
{
	/* Create relevant typedefs for multidimensional arrays */
	SgArrayType *arr_type = isSgArrayType(arr_name->get_type());
	if(!arr_type)
		return NULL;
	
	SgType *base = arr_type->get_base_type();
	std::vector<SgExpression*> exp_list = SageInterface::get_C_array_dimensions(arr_type, *arr_name);

	std::string typedef_name = "_narray_" + arr_name->get_name().getString();
	SgTypedefDeclaration *arr_td = SageBuilder::buildTypedefDeclaration(typedef_name, base, parentScope);
	SageInterface::appendStatement(arr_td, bb_new);	

	/* Obtain the absolute base type of the array */
	SgType *abs_base = base->findBaseType();
	
	/* Find the appropriate number of bytes (return this expression so it can be used in the memcpy) */
       	SgMultiplyOp *num_bytes = SageBuilder::buildMultiplyOp(SageBuilder::buildSizeOfOp(abs_base), exp_list[1]);
	for(size_t i = 2; i < exp_list.size(); i++)
		num_bytes = SageBuilder::buildMultiplyOp(num_bytes, exp_list[i]);
	
	/* Declare a pointer which will be used on the device: _narray_i *d_arr_name; */
	std::string d_arr_ptr_name = "d_" + arr_name->get_name().getString();
	SgVariableDeclaration *d_arr_ptr = SageBuilder::buildVariableDeclaration(
									d_arr_ptr_name,
									SageBuilder::buildPointerType(arr_td->get_type()),
									NULL,
									parentScope );
	std::string malloc_string = "\n    cudaMalloc((void **) &" + d_arr_ptr_name + ", " + num_bytes->unparseToString() + ");";
	std::string memcpy_string = "\n    cudaMemcpy(" + d_arr_ptr_name + ", " + arr_name->get_name().getString() + ", " + num_bytes->unparseToString() + ", cudaMemcpyHostToDevice);\n";
	SageInterface::addTextForUnparser(d_arr_ptr, malloc_string + memcpy_string, AstUnparseAttribute::e_after);

	/* Make the call to cudaMalloc() and cudaMemcpy() right before the loop nest (before the serial loop in the ECS case) */
	SageInterface::appendStatement(d_arr_ptr, bb_new);

	/* Return the bytes expression so it can be used to memcpy back to host */
	return num_bytes;

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
		
	/* Create the function declaration and apply function modifier to make it a global kernel function */
	SgFunctionDeclaration *kernel_fn = SageBuilder::buildDefiningFunctionDeclaration(kernel_name, SageBuilder::buildVoidType(), kernel_params, globalScope);
	SgFunctionModifier &kernel_mod = kernel_fn->get_functionModifier();
	//kernel_mod.setCudaGlobalFunction();
	kernel_mod.setCudaKernel();

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

		/* Create declaration for all of iter vars in the nested loops which will be run on the GPU */
		for(size_t i = 3; i < inner_loops.size(); i++)
		{
			SgForStatement *loop = isSgForStatement(inner_loops[i]);
			SgStatementPtrList &init_list = loop->get_init_stmt();
			SgStatement *init = init_list.front();

			SgExpression *var;
			if(SageInterface::isAssignmentStatement(init, &var))
			{
				SgVarRefExp *var_ref = isSgVarRefExp(var);
				SgVariableDeclaration *var_decl = SageBuilder::buildVariableDeclaration(var_ref->get_symbol()->get_name(), SageBuilder::buildIntType(), NULL, kernel_body);
				SageInterface::appendStatement(var_decl, kernel_body);
			}

		}
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
	//SageInterface::prependStatement(kernel_fn, globalScope);
	SgStatement *first_stmt = SageInterface::getFirstStatement(globalScope);
	if(first_stmt)
		SageInterface::insertStatementBefore(first_stmt, kernel_fn);
	else
		SageInterface::prependStatement(kernel_fn, globalScope);

}

