/* Implementation of parallelism extraction */
#include "./parallel.hpp"


/* Extract any parallelism in the loop */
bool extractParallelism(SgForStatement *loop_nest, SgGlobal *globalScope, int &nest_id, bool &ecs_fn_flag)
{
	/* Obtain attributes of loop nest */
	LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();
	std::list<SgExpression*> loop_bound_vec = attr->get_bound_vec();
	std::list<std::string> loop_symb_vec = attr->get_symb_vec();
	int loop_nest_size = attr->get_nest_size();
	
	/* Obtain body of loop */
	Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
	SgBasicBlock *body = isSgBasicBlock(isSgForStatement(inner_loops[loop_nest_size - 1])->get_loop_body());

	/* Obtain dependence graph of body */
	Graph *dep_graph = getDependencyGraph(body);
	
	/* Find strongly connected components in graph */
	std::list<std::list<int>> scc_list = dep_graph->getSCCs();

	/* Go thru SCC list and obtain SCCs with more than one node */
	std::list<std::list<int>> multi_scc_list;
	for(auto scc_it = scc_list.begin(); scc_it != scc_list.end(); scc_it++)
		if((*scc_it).size() > 1)
			multi_scc_list.push_back(*scc_it);
	
	/* If every SCC contains only one node, then loop is a candidate for loop fission */ 
	if(multi_scc_list.empty())
		return loopFission(loop_nest, nest_id, globalScope);

	/* Otherwise, it is a candidate for extended cycle shrinking */
	return extendedCycleShrink(loop_nest, scc_list, dep_graph->getADJ(), globalScope, nest_id, ecs_fn_flag);

}


Graph * getDependencyGraph(SgBasicBlock *body)
{
	/* Obtain list of statements in the body */
	SgStatementPtrList &stmts = body->get_statements();

	/* Create graph */
	Graph *g = new Graph(stmts.size());

	/* List of list of set of read/write vars 
	   
	   dep_var_list[]       -->   Get particular statement
	   dep_var_list[][]     -->   Get read/write (0 = read, 1 = write)
	   dep_var_list[][][]   -->   Get var
	 
	*/
	std::vector<std::vector<std::set<SgInitializedName*>>> dep_var_list; 

	/* Go thru each statement and collect read/write references */
	for(long unsigned int i = 0; i < stmts.size(); i++)
	{
		SgStatement *s = stmts[i];

		/* Get read/write vars in this statement */
		std::set<SgInitializedName*> read_vars, write_vars;
		SageInterface::collectReadWriteVariables(s, read_vars, write_vars);

		/* Will hold read/write vars for this statement */
		std::vector<std::set<SgInitializedName*>> stmt_dep_vars;
		stmt_dep_vars.push_back(read_vars);
		stmt_dep_vars.push_back(write_vars);
		
		/* Append stmt_dep_vars to main dep_var_list */
		dep_var_list.push_back(stmt_dep_vars);
	}
	
	/* Go thru each statement again but now check for true/flow dependencies */
	for(size_t i = 0; i < dep_var_list.size(); i++)
	{
		std::vector<std::set<SgInitializedName*>> stmt_dep_vars = dep_var_list[i];
		std::set<SgInitializedName*> write_vars = stmt_dep_vars[1];
		std::set<SgInitializedName*> read_vars = stmt_dep_vars[0];

		/* Go through each write var, make sure its an array var, and check for flow deps */
		for(auto w_it = write_vars.begin(); w_it != write_vars.end(); w_it++)
		{
			/* Skip non-array vars */
			if(!isSgArrayType((*w_it)->get_type()))
				continue;

			//std::cout << "W: " << (*w_it)->get_name().getString() << std::endl;	
			/* Check every succeeding statement for a read of the current array var */
			for(size_t j = i; j < dep_var_list.size(); j++)
			{
				std::vector<std::set<SgInitializedName*>> next_stmt_dep_vars = dep_var_list[j];
				std::set<SgInitializedName*> next_read_vars = next_stmt_dep_vars[0];

				/* Check for same reference and add to graph if so */
				for(auto r_it = next_read_vars.begin(); r_it != next_read_vars.end(); r_it++)	
				{
					//std::cout << "R: " << (*r_it)->get_name().getString() << std::endl;
					if( (*w_it) == (*r_it) )
					{
						//std::cout << "Adding edge (" << i << "," << j << ")" << std::endl;
						g->addEdge(i,j);
						
					}
				}

			}
		}

		/* Now, go thru each read var, make sure its an array var, and check for anti deps */
		for(auto r_it = read_vars.begin(); r_it != read_vars.end(); r_it++)
		{
			/* Skip non-array vars */
			if(!isSgArrayType((*r_it)->get_type()))
				continue;

			/* Check every succeeding statement for a write of the current array var */
			for(size_t j = i; j < dep_var_list.size(); j++)
			{
				std::vector<std::set<SgInitializedName*>> next_stmt_dep_vars = dep_var_list[j];
				std::set<SgInitializedName*> next_write_vars = next_stmt_dep_vars[1];

				/* Check for same reference and add to graph if so */
				for(auto w_it = next_write_vars.begin(); w_it != next_write_vars.end(); w_it++)	
					if( (*r_it) == (*w_it) )
						g->addEdge(j, i);
			}
		}


	}

	return g;

}


/* Perform loop fission */
bool loopFission(SgForStatement *loop_nest, int &nest_id, SgGlobal *globalScope)
{
	/* Obtain attributes of loop nest */
	LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));
	int loop_nest_size = attr->get_nest_size();
	
	/* Obtain statements in body of loop */
	Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
	SgBasicBlock *body = isSgBasicBlock(isSgForStatement(inner_loops[loop_nest_size - 1])->get_loop_body());
	SgStatementPtrList &stmts = body->get_statements();

	/* Build the new basic block which will hold the new loops */
	SgBasicBlock *bb = SageBuilder::buildBasicBlock();
	bb->set_parent(loop_nest->get_parent());

	/* Only perform fission if every statement is an assign and the LHS is an array reference (to avoid any additional variables or declarations) */
	for(unsigned long i = 0; i < stmts.size(); i++)
	{
		SgExprStatement *s = isSgExprStatement(stmts[i]);
		if(!s)
			return false;

		/* Obtain assignment expression */
		SgExpression *expr = s->get_expression();

		/* Binary Op Case */
		if(isSgBinaryOp(expr))
		{
			SgExpression *lhs = isSgBinaryOp(expr)->get_lhs_operand();
			SgExpression *rhs = isSgBinaryOp(expr)->get_rhs_operand();

			/* If the LHS of the binary op is not an arr ref, do not perform fission */
			if(!isSgPntrArrRefExp(lhs))
				return false;

			/* If rhs contains reference to lhs array, check to see if dependent (using dependency tests in ../dependency/dependency.cpp) */
			SgInitializedName *lhs_name = SageInterface::convertRefToInitializedName(isSgPntrArrRefExp(lhs));
			std::vector<SgExpression*> *lhs_dim = new std::vector<SgExpression*>;
			SageInterface::isArrayReference(isSgPntrArrRefExp(lhs), NULL, &lhs_dim);
			int test_flag = 0;

			Rose_STL_Container<SgNode*> rhs_arr_refs = NodeQuery::querySubTree(rhs, V_SgPntrArrRefExp);
			for(auto arr_it = rhs_arr_refs.begin(); arr_it != rhs_arr_refs.end(); arr_it++)
			{
				if( SageInterface::convertRefToInitializedName(isSgPntrArrRefExp(*arr_it)) != lhs_name )
					continue;

				/* Obtain the subscript expressions for the rhs array reference */
				std::vector<SgExpression*> *rhs_dim = new std::vector<SgExpression*>;
				SageInterface::isArrayReference(isSgPntrArrRefExp(*arr_it), NULL, &rhs_dim);

				/* Perform the tests */
				test_flag = ZIVTest(*lhs_dim, *rhs_dim);
				if(test_flag == 2)
					return false;
				else if(test_flag == 0)
					continue;
				
				test_flag = GCDTest(*lhs_dim, *rhs_dim, attr);
				if(test_flag == 2)
					return false;
				else if(test_flag == 0)
					continue;

				test_flag = banerjeeTest(*lhs_dim, *rhs_dim, attr);
				if(test_flag)
					return false;
				else 
					continue;

			}

		}

		/* If operand of unary op is not an arr ref, do not perform fission */
		if(isSgUnaryOp(expr))
			if(!isSgPntrArrRefExp(isSgUnaryOp(expr)->get_operand()))
				return false;

		/* If we get here, then fission can be performed for this statement */

		/* Copy the for loop */
		SgForStatement *new_loop_nest = isSgForStatement(SageInterface::copyStatement(loop_nest));

		/* Replace the body with just the single statement */
		Rose_STL_Container<SgNode*> new_inner_loops = NodeQuery::querySubTree(new_loop_nest, V_SgForStatement);
		SgForStatement *inner_most_loop = isSgForStatement(isSgForStatement(new_inner_loops[loop_nest_size - 1]));
		inner_most_loop->set_loop_body(SageBuilder::buildBasicBlock(s));

		/* Append the new_loop_nest to the created bb */
		new_loop_nest->set_parent(loop_nest->get_parent());
		SageInterface::appendStatement(new_loop_nest, bb);

		/* Make call to kernel generation for new_loop_nest (should be same case as simple code gen) */
		kernelCodeGenSimple(new_loop_nest, globalScope, nest_id);

	}

	/* If we get here, we have successfully created loops for all of the statements in the main loop nest, so now replace loop_nest with the bb */
	isSgStatement(loop_nest->get_parent())->replace_statement(loop_nest, bb);

	return true;
}


/* Perform extended cycle shrinking */
bool extendedCycleShrink(SgForStatement *loop_nest, std::list<std::list<int>> scc_list, std::list<int> *adj_list, SgGlobal *globalScope, int &nest_id, bool &ecs_fn_flag)
{
	/* Obtain attributes from the loop nest */
	LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();
	std::list<SgExpression*> loop_bound_vec = attr->get_bound_vec();
	int loop_nest_size = attr->get_nest_size();
	
	/* Obtain loop body and statements */
	Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
	SgBasicBlock *body = isSgBasicBlock(isSgForStatement(inner_loops[loop_nest_size - 1])->get_loop_body());
	SgStatementPtrList &stmts = body->get_statements();
	
	/* Create a basic block which will hold all of the transformed loops (will ultimately replace loop_nest) */
	SgBasicBlock *bb_new = SageBuilder::buildBasicBlock();

	
	/*
	********************************************************************************
	       Create the min/max function defns -- Will be used in ECS algorithm
	********************************************************************************
	*/
	if(!ecs_fn_flag)
	{
		createECSFn("ecsMaxFn", globalScope);
		createECSFn("ecsMinFn", globalScope);
		ecs_fn_flag = true;
	}


	
	/* Go through each SCC and try to perform ECS on it */
	for(auto scc_it = scc_list.begin(); scc_it != scc_list.end(); scc_it++)
	{
		/* Obtain the SCC */
		std::list<int> scc = *scc_it;

		/* 
		*********************************** 
		       Check for Dep. Cycles
		***********************************
	        */

		/* Check to make sure that the members of the SCC only include each other in their adj list */
		for(auto v_it = scc.begin(); v_it != scc.end(); v_it++)
		{
			/* Get the adj list for the current vertex */
			std::list<int> curr_adj_list = adj_list[*v_it];

			/* Check the adj list for the vertex.  If it includes a member not in the SCC, then return false */
			for(auto adj_it = curr_adj_list.begin(); adj_it != curr_adj_list.end(); adj_it++)
				if( std::find(scc.begin(), scc.end(), *adj_it) == scc.end() )
					return false;
		}

		/* Move on to next SCC if this one only contains one component */
		if(scc.size() <= 1)
			continue;


		/*
		************************************************
		       Obtain Dependence Distance Vectors
		************************************************
		*/
		
		/* Array to hold dependence distance vectors for each loop_iter var */
		std::vector<int> *ddv_arr = new std::vector<int>[loop_iter_vec.size()];

		/* Create a temp basic block for the statements in the SCC */
		SgBasicBlock *bb_temp = SageBuilder::buildBasicBlock();
		bb_temp->set_parent(loop_nest->get_parent());
		std::vector<SgStatement*> scc_stmts;

		/* Sort the SCC so that statements get appended in the right order */
		scc.sort();

		/* For each statement in the SCC, collect the dependence references */
		for(auto v_it = scc.begin(); v_it != scc.end(); v_it++)
			scc_stmts.push_back(stmts[*v_it]);
		
		/* Set the scc_stmts as the body of bb_temp */
		SageInterface::appendStatementList(scc_stmts, bb_temp);

		/* Now, find all read/write stmts in bb_temp */
		std::vector<SgNode*> read_stmts, write_stmts;
		SageInterface::collectReadWriteRefs(bb_temp, read_stmts, write_stmts);

		/* Go through each write_stmt */
		for(size_t w_it = 0; w_it < write_stmts.size(); w_it++)
		{
			/* Skip any non-array references */
			SgPntrArrRefExp *w_arr_ref = isSgPntrArrRefExp(write_stmts[w_it]);
			if(!w_arr_ref)
				continue;

			/* Get name of array */
			SgInitializedName *w_name = SageInterface::convertRefToInitializedName(w_arr_ref);

			/* Go through the writes again and make sure we are not writing to the same array (to be conservative) */
			for(size_t w_it_next = w_it + 1; w_it_next < write_stmts.size(); w_it_next++)
			{
				/* Again, skip any non-array references */
				SgPntrArrRefExp *w_next_arr_ref = isSgPntrArrRefExp(write_stmts[w_it_next]);
				if(!w_next_arr_ref)
					continue;

				/* If we write to the same array, return false (to be conservative) */
				if(SageInterface::convertRefToInitializedName(w_next_arr_ref) == w_name)
					return false;
			}

			/* Go through the reads now */
			for(size_t r_it = 0; r_it < read_stmts.size(); r_it++)
			{
				/* Skip non-array references */
				SgPntrArrRefExp *r_arr_ref = isSgPntrArrRefExp(read_stmts[r_it]);
				if(!r_arr_ref)
					continue;

				/* Skip array references that do not deal with the write array */
				if(SageInterface::convertRefToInitializedName(r_arr_ref) != w_name)
					continue;

				/* Compute the DDV for this particular reference */
				if(!computeDDV(w_arr_ref, r_arr_ref, ddv_arr, attr))
					return false;

			}
			
		}

		/* If we get here, then all of the DDVs have been computed for this SCC */


		/*
		*****************************
		       Obtain Loop DDV
		*****************************
		*/
		int *loop_ddv = new int[loop_iter_vec.size()];

		/* Rules for loop_ddv:

		  loop_ddv[i] = 0                            if ddv_arr[i][j] is 0 for some j, or if ddv_arr[i][j] and ddv_arr[i][k] have opposite signs for some j,k
		  loop_ddv[i] = min(abs(ddv_arr[i][j]))      if sign(ddv_arr[i][j]) is positive for all j
		  loop_ddv[i] = -min(abs(ddv_arr[i][j]))     if sign(ddv_arr[i][j]) is negative for all j 

		*/
		for(size_t i = 0; i < loop_iter_vec.size(); i++)
		{
			std::vector<int> iter_ddv = ddv_arr[i];
			std::sort(iter_ddv.begin(), iter_ddv.end());
		
			/* Check for any zeros */
			if(std::find(iter_ddv.begin(), iter_ddv.end(), 0) != iter_ddv.end())
			{
				//loop_ddv[i] = 0;
				//continue;
				// If any are zero, ecs cannot be performed, so just return false
				return false;
			}

			/* Check for any sign changes: 1 = pos, -1 = neg */
			int sign_front = getSign(iter_ddv.front());
			int sign_back = getSign(iter_ddv.back());

			if( sign_front != sign_back )
			{
				//loop_ddv[i] = 0;
				//continue;
				// Same here, loop_ddv[i] will be zero so just return false
				return false;
			}


			/* If the sign is positive, take the smallest element in the vector (i.e. front()) */
			if(sign_front == 1)
				loop_ddv[i] = iter_ddv.front();
			/* Otherwise, if the sign is negative, take the largest element (since it has the smallest abs value) */
			else
				loop_ddv[i] = iter_ddv.back();

		}

		/*
		*********************************
		       Create the New Loop              
		*********************************
		*/
		
		/* Convert bounds expression to INT, return false if not possible */
		std::vector<int> bound_vec;
		for(auto b_it = loop_bound_vec.begin(); b_it != loop_bound_vec.end(); b_it++)
			if(isSgIntVal(*b_it))
				bound_vec.push_back( isSgIntVal(*b_it)->get_value() );
			else
				return false;

		/*
		********************************************
		       Step 1: Outer-most Serial Loop
		********************************************
		*/
		int scc_index = std::distance(scc_list.begin(), scc_it);
		std::string ecs_serial_index_name = "ecs_serial_index_" + std::to_string(scc_index);
		
		/* Introduce a serial loop whose index increases from 1 to min(ceil(abs(bound_vec[i]/loop_ddv[i]))) with stride of 1 */
		SgVariableDeclaration *serial_init = SageBuilder::buildVariableDeclaration(
									ecs_serial_index_name, 
									SageBuilder::buildIntType(), 
									SageBuilder::buildAssignInitializer(SageBuilder::buildIntVal(1)),
			       						loop_nest );
		
		SgVarRefExp *ecs_serial_index = SageBuilder::buildVarRefExp(ecs_serial_index_name, loop_nest);

		int min_val = INT_MAX;
		for(size_t i = 0; i < bound_vec.size(); i++)
		{
			if(loop_ddv[i] == 0)
				continue;

			int intermed = getCeil(getAbs( (double)bound_vec[i] / loop_ddv[i] ));
			if(intermed < min_val)
				min_val = intermed;
		}

		SgExprStatement *serial_test = SageBuilder::buildExprStatement(
									SageBuilder::buildLessOrEqualOp(
												/*SageBuilder::buildVarRefExp(*/ecs_serial_index,
												SageBuilder::buildIntVal(min_val) ) );

		SgExpression *serial_stride = SageBuilder::buildPlusPlusOp(
								/*SageBuilder::buildVarRefExp(*/ecs_serial_index );

		SgBasicBlock *serial_body = SageBuilder::buildBasicBlock();

		SgForStatement *serial_loop = SageBuilder::buildForStatement(serial_init, serial_test, serial_stride, serial_body);
		serial_loop->set_parent(loop_nest->get_parent());
		/*
		*************************************************************
		       Step 2: Inner Serial Loop to Find Proper Bounds
		*************************************************************
		*/
		
		/* Store the start[] array internally */
		SgExpression **start = new SgExpression*[loop_iter_vec.size()];
		for(size_t i = 0; i < loop_iter_vec.size(); i++)
		{
			if(loop_ddv[i] >= 0)
				start[i] = SageBuilder::buildAddOp(
								SageBuilder::buildIntVal(1),
								SageBuilder::buildMultiplyOp(
											SageBuilder::buildSubtractOp(
														/*SageBuilder::buildVarRefExp(*/ecs_serial_index,
													        SageBuilder::buildIntVal(1) ),
										        SageBuilder::buildIntVal(loop_ddv[i]) ) );
			else
				start[i] = SageBuilder::buildAddOp(
								SageBuilder::buildIntVal(bound_vec[i]),
								SageBuilder::buildMultiplyOp(
											SageBuilder::buildSubtractOp(
														/*SageBuilder::buildVarRefExp(*/ecs_serial_index,
													        SageBuilder::buildIntVal(1) ),
										        SageBuilder::buildIntVal(loop_ddv[i]) ) );
		}

		/*
		*******************************************************
		       Step 3: Introduce the Parallel Statements
		*******************************************************
		*/
		SgBasicBlock *serial_inner_bb = SageBuilder::buildBasicBlock();
		serial_inner_bb->set_parent(loop_nest->get_parent());
		std::vector<SgForStatement*> parallel_loops;
		std::vector<SgBasicBlock*> parallel_loop_bbs;
		for(size_t i = 0; i < loop_iter_vec.size(); i++)
		{
			/* If for some reason we missed loop_ddv[i] being zero, return false here */
			if(loop_ddv[i] == 0)
				return false;

			/* Do some iterator arith since loop_iter_vec is a std::list */
			auto loop_iter_it = loop_iter_vec.begin();
			std::advance(loop_iter_it, i);

			SgVarRefExp *curr_iter_var = SageBuilder::buildVarRefExp(*loop_iter_it);
			SgExprStatement *curr_init = SageBuilder::buildExprStatement(
										SageBuilder::buildAssignOp(
													curr_iter_var,
													start[i] ) );
			SgExprStatement *curr_test;
			SgExprStatement *curr_stride;

			/* loop_iter_vec[i] going from start[i] to min{start[i] + loop_ddv[i] - 1, bound_vec[i]} with stride 1 */
			if(loop_ddv[i] > 0)
			{
				/* Make a function call to ecsMinFn(), which was created */
				std::vector<SgExpression*> expr_list;
				expr_list.push_back(
						SageBuilder::buildAddOp(
								start[i],
								SageBuilder::buildSubtractOp(
											SageBuilder::buildIntVal(loop_ddv[i]),
											SageBuilder::buildIntVal(1) ) ) );	
				
				expr_list.push_back(SageBuilder::buildIntVal(bound_vec[i])); 
				
				SgExprStatement *call = SageBuilder::buildFunctionCallStmt(
											"ecsMinFn", 
											SageBuilder::buildIntType(),
											SageBuilder::buildExprListExp(expr_list),
					       						body );
	
				/* Make the correct test */
				curr_test = SageBuilder::buildExprStatement(
									SageBuilder::buildLessOrEqualOp(
												curr_iter_var,
												call->get_expression() ) );
				/* Stride is +1 */
				curr_stride = SageBuilder::buildExprStatement(
									SageBuilder::buildPlusPlusOp(curr_iter_var) );

			}
			/* loop_iter_vec[i] going from start[i] to max{start[i] + loop_ddv[i] + 1, 1} with stride -1 */
			else
			{
				/* Make a function call to ecsMaxFn(), which was created */
				std::vector<SgExpression*> expr_list;
				expr_list.push_back(
						SageBuilder::buildAddOp(
								start[i],
								SageBuilder::buildAddOp(
											SageBuilder::buildIntVal(loop_ddv[i]),
											SageBuilder::buildIntVal(1) ) ) );	
				
				expr_list.push_back(SageBuilder::buildIntVal(1)); 
				
				SgExprStatement *call = SageBuilder::buildFunctionCallStmt(
											"ecsMaxFn", 
											SageBuilder::buildIntType(),
											SageBuilder::buildExprListExp(expr_list),
					       						body );

				/* Make the correct test */
				curr_test = SageBuilder::buildExprStatement(
									SageBuilder::buildLessOrEqualOp(
												curr_iter_var,
												call->get_expression() ) );
				/* Stride is -1 */
				curr_stride = SageBuilder::buildExprStatement(
									SageBuilder::buildMinusMinusOp(curr_iter_var) );

			
			}

			/* Create the curr_loop body, and the loop itself */
			SgBasicBlock *curr_body = SageBuilder::buildBasicBlock();
			SgForStatement *curr_loop = SageBuilder::buildForStatement(curr_init, curr_test, curr_stride->get_expression(), curr_body);
			curr_loop->set_parent(loop_nest->get_parent());
			/*
			    Then, introduce n-1 for loops with loop_iter_vec[j] where i != j
			    If j < i, loop_ddv[i] > 0 bounds is start[i] + loop_ddv[i] to bound_vec[i]
			              loop_ddv[i] < 0 bounds is start[i] + loop_ddv[i] to 1 
			    If j > i, bounds is start[i] to bound_vec[i]
		  	
			    The body of this loop nest should be the statements in the SCC
			*/
			for(size_t j = 0; j < loop_iter_vec.size(); j++)
			{
				if(j == i)
					continue;

				/* Obtain current inner-most loop */
				Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(curr_loop, V_SgForStatement);
				SgForStatement *inner_loop = isSgForStatement(inner_loops.back());

				/* Will set this as the body of inner_loop once done */
				SgForStatement *new_inner_loop;
				
				/* Iterator arith to access loop_iter_vec[j] */
				auto inner_loop_iter_it = loop_iter_vec.begin();
				std::advance(inner_loop_iter_it, j);

				SgVarRefExp *inner_iter_var = SageBuilder::buildVarRefExp(*inner_loop_iter_it);
				SgExprStatement *inner_init, *inner_test; 
				SgExpression *inner_stride = SageBuilder::buildPlusPlusOp(inner_iter_var);
				SgBasicBlock *inner_body = SageBuilder::buildBasicBlock();
				
				if(j < i)
				{
					inner_init = SageBuilder::buildExprStatement(
										SageBuilder::buildAssignOp(
													inner_iter_var, 
													SageBuilder::buildAddOp(
															start[j],
															SageBuilder::buildIntVal(loop_ddv[j]) ) ) );
					if(loop_ddv[j] > 0)
						inner_test = SageBuilder::buildExprStatement(
											SageBuilder::buildLessOrEqualOp(
														inner_iter_var,
														SageBuilder::buildIntVal(bound_vec[j]) ) );
					else
						inner_test = SageBuilder::buildExprStatement(
											SageBuilder::buildLessOrEqualOp(
														inner_iter_var,
														SageBuilder::buildIntVal(1) ) );
				}

				if(j > i)
				{
					inner_init = SageBuilder::buildExprStatement(
										SageBuilder::buildAssignOp(
													inner_iter_var, 
													start[j]) );
					inner_test = SageBuilder::buildExprStatement(
										SageBuilder::buildLessOrEqualOp(
													inner_iter_var,
													SageBuilder::buildIntVal(bound_vec[j]) ) );

				}

				new_inner_loop = SageBuilder::buildForStatement(inner_init, inner_test, inner_stride, inner_body);
				
				/* Set the body of the old inner most loop as the new inner most loop */
				inner_loop->set_loop_body(new_inner_loop);
			}

			/* Now, set the body of the inner most loop as the kernel call for the SCC statements */
			Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(curr_loop, V_SgForStatement);
			for(auto l_it = inner_loops.begin(); l_it != inner_loops.end(); l_it++)
				(*l_it)->set_parent(serial_loop);
				
			SgForStatement *inner_loop = isSgForStatement(inner_loops.back());
			SgBasicBlock *inner_bb = SageBuilder::buildBasicBlock();
			inner_bb->set_parent(loop_nest->get_parent());	
			for(auto v_it = scc.begin(); v_it != scc.end(); v_it++)
				SageInterface::appendStatement(stmts[*v_it], inner_bb);  // TODO: Kernel call here 
			inner_loop->set_loop_body(inner_bb);

			
			//inner_loop->set_parent(serial_loop);
			/* Append the current loop to the serial_inner_bb */
			SageInterface::appendStatement(curr_loop, serial_inner_bb);  //-- This works for ECS without kernel gen, commenting out to test out kernel gen	
			parallel_loop_bbs.push_back(inner_bb);	
			parallel_loops.push_back(curr_loop);

		}

		/* Set body of serial_loop to serial_inner_bb */
		serial_loop->set_loop_body(serial_inner_bb);

		/* Make call to SageInterface::fixVariableReferences() due to the buildVarRefExp */
		SageInterface::fixVariableReferences(serial_loop);
		
		SgBasicBlock *kernel_bb = isSgBasicBlock(kernelCodeGenECS(serial_loop, parallel_loops, parallel_loop_bbs, nest_id, globalScope));

		/*
		******************************************
		       Append Serial Loop to New BB       
		******************************************
		*/
		//SageInterface::appendStatement(serial_loop, bb_new);  -- This works for ECS without kernel gen, commenting out to test out kernel gen
		SageInterface::appendStatement(kernel_bb, bb_new);
	}

	/*
	*********************************************************
	       Replace old loop nest with the serial loops
	*********************************************************
	*/
	isSgStatement(loop_nest->get_parent())->replace_statement(loop_nest, bb_new);	
	

	return true;
}



/* Compute the DDV for a particular reference */
bool computeDDV(SgPntrArrRefExp *w_arr_ref, SgPntrArrRefExp *r_arr_ref, std::vector<int> *ddv_arr, LoopNestAttribute *attr)
{
	/* Obtain attributes */
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();
	std::list<std::string> loop_symb_vec = attr->get_symb_vec();
	int vec_size = loop_iter_vec.size() + loop_symb_vec.size();
	
	/* Obtain the dimension info */
	std::vector<SgExpression*> *w_dim = new std::vector<SgExpression*>;
	std::vector<SgExpression*> *r_dim = new std::vector<SgExpression*>;
	SageInterface::isArrayReference(w_arr_ref, NULL, &w_dim);
	SageInterface::isArrayReference(r_arr_ref, NULL, &r_dim);

	/* Make sure the dimensions are of equal size */
	if((*w_dim).size() != (*r_dim).size())
		return false;

	/* Only handle SIV cases (i.e. each subscript contains reference to at most one var, and that var is the same for both array references) */
	for(size_t i = 0; i < (*w_dim).size(); i++)
	{
		/* Obtain variable references in the two subscripts */
		Rose_STL_Container<SgNode*> w_var_refs = NodeQuery::querySubTree((*w_dim)[i], V_SgVarRefExp);
		Rose_STL_Container<SgNode*> r_var_refs = NodeQuery::querySubTree((*r_dim)[i], V_SgVarRefExp);

		/* If either has reference to more than one variable, return false */
		if( (w_var_refs.size() > 1) || (r_var_refs.size() > 1) )
			return false;
				
		/* If both have no reference to a variable, go to next subscript */
		if( w_var_refs.empty() && r_var_refs.empty() )
			continue;
		
		/* If only one of the references has a reference to a variable, return false */
		if( w_var_refs.size() != r_var_refs.size() )
			return false;


		/* If neither of those conditions are met, then we have potential for parallelism */
		
		/* Variable to hold which loop index variable is being referenced */ 
		int ddv_loop_index = -1;

		/* If both only contain one reference, make sure the same variable is being referenced */
		if( (w_var_refs.size() == 1) && (r_var_refs.size() == 1) )
		{
			std::string w_var = isSgVarRefExp(w_var_refs[0])->get_symbol()->get_name().getString();
			std::string r_var = isSgVarRefExp(r_var_refs[0])->get_symbol()->get_name().getString();

			if(w_var != r_var)
				return false;

			/* Get the index of variable being referenced */
			auto iter_it = std::find(loop_iter_vec.begin(), loop_iter_vec.end(), w_var);

			/* If it is not an iter var, return false to be conservative */
			if(iter_it == loop_iter_vec.end())
				return false;

			ddv_loop_index = std::distance(loop_iter_vec.begin(), iter_it);
		}

		/* If, for some reason, ddv_loop_index remained negative, return false */
		if(ddv_loop_index < 0)
			return false;


		/* Now, actually obtain the DDV */	
		
		/* Extract coefficients using function defined in ../dependency/dependency.cpp -- Note: We ignore data about symb_vec to remain conservative */
		std::vector<int> w_coeff(vec_size), r_coeff(vec_size);
		int w_const = 0, r_const = 0;

		/* If coeff extraction fails, return false */
		if(!extractCoeff((*w_dim)[i], w_coeff, w_const, attr))
			return false;
		if(!extractCoeff((*r_dim)[i], r_coeff, r_const, attr))
			return false;

		/* Obtain the distance:  Deal with cases where we can obtain distance (i.e. a1 = b1)
		 	w:   a1*i + a0
			r:   b1*i + b0

		   distance = a0 - b0
		*/

		if(w_coeff[ddv_loop_index] != r_coeff[ddv_loop_index])
			return false;

		/* This should not happen, since GCD should cover it, but just to be safe */
		if( ( (w_const - r_const) % w_coeff[ddv_loop_index] ) != 0 )
			return false;

		int distance = (w_const - r_const) / w_coeff[ddv_loop_index];

		/* Add this distance to the relevant location in the ddv_arr_temp */
		ddv_arr[ddv_loop_index].push_back(distance);
	}

	/* If we get here, then we were able to obtain DDVs for all of the subscripts, so return true */
       	return 	true;
}

/* Helper function to get the sign of a number */
int getSign(int num)
{
	if(num >= 0)
		return 1;
	else
		return -1;
}

/* Helper function to get absolute value of a number */
double getAbs(double num)
{
	if(num >= 0)
		return num;
	else
		return -num;

}

/* Helper function to get ceiling of a number */
int getCeil(double num)
{
	return (int)std::ceil(num);		
}

void createECSFn(std::string name, SgGlobal *globalScope)
{
	/* Parameter list (will always compare INT to INT) */ 
	SgName val1 = "val1", val2 = "val2";
	SgType *ref_type1 = SageBuilder::buildIntType();
	SgType *ref_type2 = SageBuilder::buildIntType();
	SgInitializedName *val1_init = SageBuilder::buildInitializedName(val1, ref_type1);
	SgInitializedName *val2_init = SageBuilder::buildInitializedName(val2, ref_type2);
	SgFunctionParameterList *param_list = SageBuilder::buildFunctionParameterList();
	SageInterface::appendArg(param_list, val1_init); 
	SageInterface::appendArg(param_list, val2_init);

	/* Function declaration */
	SgName fn_name = name;
	SgFunctionDeclaration *fn = SageBuilder::buildDefiningFunctionDeclaration(fn_name, SageBuilder::buildIntType(), param_list, globalScope);
	SgBasicBlock *fn_body = fn->get_definition()->get_body();
	
	/* Set as CUDA function so it can be called by device and host */
	SgFunctionModifier &fn_mod = fn->get_functionModifier();
	fn_mod.setCudaDevice();
	fn_mod.setCudaHost();

	/* Create statements in body of functions */
	SgVarRefExp *val1_ref = SageBuilder::buildVarRefExp(val1, fn_body);
	SgVarRefExp *val2_ref = SageBuilder::buildVarRefExp(val2, fn_body);

	/* The test is the same for min/max fns */
	SgExpression *fn_test = SageBuilder::buildGreaterThanOp(val1_ref, val2_ref);
	
	SgReturnStmt *fn_true, *fn_false;

	/* Set appropriate true/false returns depending on function name */
	if(name == "ecsMaxFn")
	{
		fn_true = SageBuilder::buildReturnStmt(val1_ref);
		fn_false = SageBuilder::buildReturnStmt(val2_ref);
	}
	else
	{
		fn_true = SageBuilder::buildReturnStmt(val2_ref);
		fn_false = SageBuilder::buildReturnStmt(val1_ref);
	}

	/* Create the if statement and append to body of function */
	SgIfStmt *if_stmt = SageBuilder::buildIfStmt(fn_test, fn_true, fn_false);
	SageInterface::prependStatement(if_stmt, fn_body);

	/* Put function definition at the top in global scope */
	//SageInterface::prependStatement(fn, globalScope);
	SgStatement *first_stmt = SageInterface::getFirstStatement(globalScope);
	if(first_stmt)
		SageInterface::insertStatementBefore(first_stmt, fn);
	else
		SageInterface::prependStatement(fn, globalScope);


}
