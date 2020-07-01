/* Implementation of parallelism extraction */
#include "./parallel.hpp"


/* Extract any parallelism in the loop */
bool extractParallelism(SgForStatement *loop_nest)
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

#if 1
	std::cout << "Strongly connected components of body:" << std::endl;
	for(auto it = scc_list.begin(); it != scc_list.end(); it++)
	{
		for(auto it2 = (*it).begin(); it2 != (*it).end(); it2++)
			std::cout << *it2 << " ";

		std::cout << std::endl;
	}
	std::cout << std::endl;
#endif

	/* Go thru SCC list and obtain SCCs with more than one node */
	std::list<std::list<int>> multi_scc_list;
	for(auto scc_it = scc_list.begin(); scc_it != scc_list.end(); scc_it++)
		if((*scc_it).size() > 1)
			multi_scc_list.push_back(*scc_it);
	
	/* If every SCC contains only one node, then loop is a candidate for loop fission */ 
	if(multi_scc_list.size() == 0)
		return loopFission(loop_nest);

	/* Otherwise, it is a candidate for extended cycle shrinking */
	//return extendedCycleShrink(loop_nest, multi_scc_list);

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
	for(long unsigned int i = 0; i < dep_var_list.size(); i++)
	{
		std::vector<std::set<SgInitializedName*>> stmt_dep_vars = dep_var_list[i];
		std::set<SgInitializedName*> write_vars = stmt_dep_vars[1];

		/* Go through each write var, make sure its an array var, and check for flow deps */
		for(auto w_it = write_vars.begin(); w_it != write_vars.end(); w_it++)
		{
			/* Skip non-array vars */
			if(!isSgArrayType((*w_it)->get_type()))
				continue;
			//std::cout << "W: " << (*w_it)->get_name().getString() << std::endl;	
			/* Check every statement for a read of the current array var */
			for(long unsigned int j = 0; j < dep_var_list.size(); j++)
			{
				std::vector<std::set<SgInitializedName*>> next_stmt_dep_vars = dep_var_list[j];
				std::set<SgInitializedName*> read_vars = next_stmt_dep_vars[0];

				/* Check for same reference and add to graph if so */
				for(auto r_it = read_vars.begin(); r_it != read_vars.end(); r_it++)	
				{
					//std::cout << "R: " << (*r_it)->get_name().getString() << std::endl;
					if( (*w_it) == (*r_it) )
					{
						g->addEdge(i,j);
						
					}
				}

			}
		}


	}

	return g;

}


/* Perform loop fusion */
bool loopFission(SgForStatement *loop_nest)
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
	//SgStatementPtrList &bb_stmts = bb->get_statements();

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
		inner_most_loop->set_loop_body(s);

		/* Append the new_loop_nest to the created bb */
		SageInterface::appendStatement(new_loop_nest, bb);

	}

	/* If we get here, we have successfully created loops for all of the statements in the main loop nest, so now replace loop_nest with the bb */
	isSgStatement(loop_nest->get_parent())->replace_statement(loop_nest, bb);

	return true;
}


/* Perform extended cycle shrinking */
bool extendedCycleShrink(SgForStatement *loop_nest, std::list<std::list<int>> scc_list)
{
	
	return true;
}

#if 0
/* Perform extended cycle shrinking on the loop nest */
bool extendedCycleShrink(SgForStatement *loop_nest)
{
	/* Obtain attributes of loop nest */
	LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();
	std::list<SgExpression*> loop_bound_vec = attr->get_bound_vec();
	std::list<std::string> loop_symb_vec = attr->get_symb_vec();
	std::list<std::list<std::list<std::vector<SgExpression*>>>> arr_dep_info = attr->get_arr_dep_info();
	int loop_nest_size = attr->get_nest_size();

	
	/* Loop thru each arr in arr_dep_info and perform cycle shrinking */
	for(auto arr_it = arr_dep_info.begin(); arr_it != arr_dep_info.end(); arr_it++)
	{
		/* Dependence Distance Vector component for this array (holds distances for each subscript*/
		//std::vector<int> ddv_comp;

		/* Obtain read and write reference lists for this particular array */
		std::list<std::list<std::vector<SgExpression*>>>::iterator ref_it = (*arr_it).begin();
		std::list<std::vector<SgExpression*>> read_refs = *ref_it;
		ref_it++;
		std::list<std::vector<SgExpression*>> write_refs = *ref_it;

		/* NEED A WAY TO FIND CYCLES IN A LOOP */	




	}

}
#endif
