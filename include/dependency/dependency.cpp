/* Implementation of dependency tests */
#include "dependency.hpp"

/* Determines if dependencies may exist, or if they definitely do not exist */
bool dependencyExists(SgForStatement *loop_nest)
{
	/* Obtain attributes of loop nest */
	LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();
	std::list<SgExpression*> loop_bound_vec = attr->get_bound_vec();
	std::list<std::string> loop_symb_vec = attr->get_symb_vec();
	int loop_nest_size = attr->get_nest_size();

	/* Obtain body of loop nest (assuming it is perfectly nested) */
	Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
	SgStatement *body = isSgForStatement(inner_loops[loop_nest_size - 1])->get_loop_body();

	/* Obtain the read/write references in the body */
	std::vector<SgNode*> read_stmts, write_stmts;
	//std::cout << "BODY:" << std::endl;
	//std::cout << body->unparseToString() << std::endl;
	SageInterface::collectReadWriteRefs(body, read_stmts, write_stmts);

	/* Collect read/write vars in the body */
	std::set<SgInitializedName*> read_vars, write_vars;
	SageInterface::collectReadWriteVariables(body, read_vars, write_vars);

	/* Copy array vars that are both read and written into a new set */
	std::set<SgInitializedName*> dep_vars;
	for(std::set<SgInitializedName*>::iterator it = write_vars.begin(); it != write_vars.end(); it++)
		if(write_vars.count(*it) || read_vars.count(*it))
			if(isSgArrayType((*it)->get_type()))
				dep_vars.insert(*it);


#if 1
	/* In order to keep hold of information about the arrays, we use a nested list:
	   
	   arr_dep_info[][][]
	   arr_dep_info[0]        -->     Contains list of list of expressions (yes, list of list of) for the first array name
	   arr_dep_info[0][0]     -->     Contains list of expressions (of the dimensions) for the first array reference of the first array name
	   arr_dep_info[0][0][0]  -->     Contains expression of first dimension of the first array reference of the first array name

           Ex:
	   	a[i][j+1][2*k] = 5;
		int x = a[1][j][k];

		arr_dep_info[0] = [[i,j+1,2*k], [1,j,k]]
		arr_dep_info[0][0] = [i,j+1,2*k]
		arr_dep_info[1] = j+1

	*/
	std::list<std::list<std::vector<SgExpression*>>> arr_dep_info;


	/* Obtain the array references */
	for(std::set<SgInitializedName*>::iterator v_it = dep_vars.begin(); v_it != dep_vars.end(); v_it++)
	{
		SgInitializedName *curr_var = *v_it;
		
		/* This will hold all references to a specific array
		   Ex: 
		       arr[i][j] = i;
		       int x = arr[i+j[0];

		   In this case,
		       arr_ref[0] = [i, j]      arr_ref[0][0] = i     arr_ref[0][1] = j
		       arr_ref[1] = [i+j, 0]    arr_ref[1][0] = i+j   arr_ref[1][1] = 0
		*/
		std::list<std::vector<SgExpression*>> arr_ref_list;

		/* Loop thru the read stmts to find references to this array */
		for(std::vector<SgNode*>::iterator r_it = read_stmts.begin(); r_it != read_stmts.end(); r_it++)
		{
			/* Make sure the stmt is an array reference expression */
			SgPntrArrRefExp *arr_ref = isSgPntrArrRefExp(*r_it);
			if(!arr_ref)
				continue;

			/* Make sure the array is referencing the current variable we are checking */
			if(SageInterface::convertRefToInitializedName(arr_ref) != (curr_var))
				continue;

			/* Obtain the list of expressions for the dimensions of the array */
			std::vector<SgExpression*> *dim_info = new std::vector<SgExpression*>;
			SageInterface::isArrayReference(arr_ref, NULL, &dim_info);
		
			/* Append the list of expressions to the parent list (i.e. list for this specific variable) */
			if(dim_info)
				arr_ref_list.push_back(*dim_info);
			
		}

		/* Do the same this for the write stmts */
		for(std::vector<SgNode*>::iterator w_it = write_stmts.begin(); w_it != write_stmts.end(); w_it++)
		{
			/* Make sure the stmt is an array reference expression */
			SgPntrArrRefExp *arr_ref = isSgPntrArrRefExp(*w_it);
			if(!arr_ref)
				continue;

			/* Make sure the array is referencing the current variable we are checking */
			if(SageInterface::convertRefToInitializedName(arr_ref) != (curr_var))
				continue;

			/* Obtain the list of expressions for the dimensions of the array */
			std::vector<SgExpression*> *dim_info = new std::vector<SgExpression*>;
			SageInterface::isArrayReference(arr_ref, NULL, &dim_info);

			/* Append the list of expressions to the parent list (i.e. list for this specific variable) */
			if(dim_info)
				arr_ref_list.push_back(*dim_info);
		}

		
		/* Now, append this info to the arr_dep_info list */
		arr_dep_info.push_back(arr_ref_list);

	}


	int n1 = 1;
	for(auto it1 = arr_dep_info.begin(); it1 != arr_dep_info.end(); it1++)
	{
		std::cout << "Arr " << n1 << ": " << std::endl;
		int n2 = 1;
		for(auto it2 = (*it1).begin(); it2 != (*it1).end(); it2++)
		{
			std::cout << "Ref " << n2 << ": " << std::endl;
			for(auto it3 = (*it2).begin(); it3 != (*it2).end(); it3++)
				std::cout << (*it3)->unparseToString() << std::endl;
			n2++;
		}
		n1++;
	}
		


	


		
#endif	
#if 0
	/* Find array references */
	std::list<SgExpression*> read_write_expr_list;
	Rose_STL_Container<SgNode*> array_refs = NodeQuery::querySubTree(body, V_SgPntrArrRefExp);
	for(auto arr_it = array_refs.begin(); arr_it != array_refs.end(); arr_it++)
	{
		SgPntrArrRefExp *expr = isSgPntrArrRefExp(*arr_it);
		SgExpression **arr_name_expr;
		if(SageInterface::isArrayReference(expr, arr_name_expr, NULL))
		{
			SgInitializedName *arr_name = SageInterface::convertRefToInitializedName(*arr_name_expr);
			if(dep_vars.count(arr_name_expr))
				read_write_expr_list.push_back(expr);
		}
	}
#endif
#if 0
	std::vector<SgNode*>::iterator it;
	std::cout << "READS:" << std::endl;
	for(it = read_stmts.begin(); it != read_stmts.end(); it++)
		std::cout << (*it)->unparseToString() << "   : " << (*it)->class_name() << std::endl;
	std::cout << "WRITES:" << std::endl;
	for(it = write_stmts.begin(); it != write_stmts.end(); it++)
		std::cout << (*it)->unparseToString() << "   : " << (*it)->class_name() << std::endl;
#endif	

	

	/* If we get here, none of the tests proved that there are no dependencies, so there may be one */
	return true;
}
