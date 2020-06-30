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
	return true;

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
