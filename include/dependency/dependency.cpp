/* Implementation of dependency tests */
#include "dependency.hpp"

/* Determines if dependencies may exist, or if they definitely do not exist */
int dependencyExists(SgForStatement *loop_nest)
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
	SageInterface::collectReadWriteRefs(body, read_stmts, write_stmts);
	
	/* Collect read/write vars in the body */
	std::set<SgInitializedName*> read_vars, write_vars;
	SageInterface::collectReadWriteVariables(body, read_vars, write_vars);

	/* Copy array vars that are written to into a new set (these have potential dependencies) */
	std::set<SgInitializedName*> dep_vars;
	for(std::set<SgInitializedName*>::iterator it = write_vars.begin(); it != write_vars.end(); it++)
		if(isSgArrayType((*it)->get_type()))
			dep_vars.insert(*it);
		
		/* If there is a write to a non-array and its scope is not the body, need to return -2 because the loop cannot be parallelized */
		else
			if((*it)->get_scope() != isSgScopeStatement(body))
				return -2;




	/* In order to keep hold of information about the arrays, we use a nested list:
	   
	   arr_dep_info[][][][]
	   arr_dep_info[0]           -->     Contains list of list of list of expressions for the first array name
	   arr_dep_info[0][0]        -->     Contains list of list of read expressions for the first array name
	   arr_dep_info[0][0][0]     -->     Contains list of expressions (of the dimensions) for the first array read reference of the first array name
	   arr_dep_info[0][0][0][0]  -->     Contains expression of first dimension of the first array read reference of the first array name

           Ex:
	   	a[i][j+1][2*k] = 5;
		int x = a[1][j][k];
		int b = a[2][2][3];

		arr_dep_info[0] = [ [[1,j,k],[2,2,3]], [[i,j+1,2*k]] ]
		arr_dep_info[0][0] = [[i,j+1,2*k], [2,2,3]]
		arr_dep_info[0][0][0] = [i,j+1,2*k]
		arr_dep_info[0][0][0][1] = j+1

	   So basically,
	        arr_dep_info[]          specifies array name
		arr_dep_info[][]        specifies read/write
		arr_dep_info[][][]      specifies reference itself
	   	arr_dep_info[][][][]    specifies expression in the subscript

	*/
	std::list<std::list<std::list<std::vector<SgExpression*>>>> arr_dep_info;


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
		std::list<std::vector<SgExpression*>> arr_read_ref_list {};
		std::list<std::vector<SgExpression*>> arr_write_ref_list {};
		std::list<std::list<std::vector<SgExpression*>>> arr_ref_list;


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
				arr_read_ref_list.push_back(*dim_info);
			
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
				arr_write_ref_list.push_back(*dim_info);
		}

		
		/* Now, append this info to the arr_dep_info list */
		arr_ref_list.push_back(arr_read_ref_list);
		arr_ref_list.push_back(arr_write_ref_list);
		arr_dep_info.push_back(arr_ref_list);

	}

	/* Add arr_dep_info as an attribute to the loop nest */
	attr->set_arr_dep_info(arr_dep_info);
	
	/* Use this flags to determine whether a dependency may exist (Initially assume that there are no deps) */
	int test_flag = 0;

	/* Loop thru each arr in arr_dep and check if dependencies exists in the references */
	for(auto arr_it = arr_dep_info.begin(); arr_it != arr_dep_info.end(); arr_it++)
	{
		/* Obtain read and write reference lists for this array */
		std::list<std::list<std::vector<SgExpression*>>>::iterator ref_it = (*arr_it).begin();
		std::list<std::vector<SgExpression*>> read_refs = *ref_it;
		ref_it++;
		std::list<std::vector<SgExpression*>> write_refs = *ref_it;
		
		/* Perform ZIV Test */
		test_flag = dependencyTests(read_refs, write_refs, attr, "ZIV");
		
		/* If we detected an error (i.e. test_flag = 2), return immediately */ 
		if(test_flag == 2)
			return test_flag;
		/* Or if we detect no dependency (i.e. test_flag = 0), continue with the rest of the arrays.  Otherwise, perform other tests */
		else if(test_flag == 0)
			continue;

		
		/* Perform GCD Test */
		test_flag = dependencyTests(read_refs, write_refs, attr, "GCD");
		if(test_flag == 2)
			return test_flag;
		else if(test_flag == 0)
			continue;
		

		/* Perform Banerjee Test */
		test_flag = dependencyTests(read_refs, write_refs, attr, "Banerjee");
		if(test_flag == 2)
			return test_flag;
		else if(test_flag == 0)
			continue;
		else 
			return 1;

	}	

	/* If we get here, either there is no dependence, or none of the tests proved that there are no dependencies, so there may be one */
	return test_flag;
}


/* Helper function to call the correct dependency test */
int dependencyTests(std::list<std::vector<SgExpression*>> read_refs, std::list<std::vector<SgExpression*>> write_refs, LoopNestAttribute *attr, std::string test_name)
{
	/* Flag used for dependency check (Initially assume no dependencies) */
	int test_flag = 0;

	/* Need to check each write against every other write, as well as against each read */
	for(long unsigned int w_it = 0; w_it != write_refs.size() - 1; w_it++)
	{	
		/* Do some iterator arithmetic */
		std::list<std::vector<SgExpression*>>::iterator curr_sub_it = std::next(write_refs.begin(), w_it);
		std::vector<SgExpression*> curr_subscripts = *curr_sub_it;

		/* Check against the writes */
		for(long unsigned int w_it_next = w_it + 1; w_it_next != write_refs.size(); w_it_next++)
		{
			std::list<std::vector<SgExpression*>>::iterator next_sub_it = std::next(write_refs.begin(), w_it_next);
			std::vector<SgExpression*> next_subscripts = *next_sub_it;
				
			/* Perform the tests:  0 --> No dep,  1 --> Possible dep,  2 --> Skip the loop */
				
			if(test_name == "ZIV")
				test_flag = ZIVTest(curr_subscripts, next_subscripts);
			else if(test_name == "GCD")
				test_flag = GCDTest(curr_subscripts, next_subscripts, attr);
			else
				test_flag = banerjeeTest(curr_subscripts, next_subscripts, attr);
			
			/* If we have detected that a dependency may exist or that the loop should be skipped, return immediately */
			if( (test_flag == 1) || (test_flag == 2) )
				return test_flag;

		}
	}

	for(auto w_it = write_refs.begin(); w_it != write_refs.end(); w_it++)
	{
		/* Get vector of array subcripts for this write ref */
		std::vector<SgExpression*> write_subscripts = *w_it;

		/* Check against the read refs */
		for(auto r_it = read_refs.begin(); r_it != read_refs.end(); r_it++)
		{
			/* Get the vector of array subscripts for this particular read reference */
			std::vector<SgExpression*> read_subscripts = *r_it;

			/* Perform the tests */
			if(test_name == "ZIV")
				test_flag = ZIVTest(write_subscripts, read_subscripts);
			else if(test_name == "GCD")
				test_flag = GCDTest(write_subscripts, read_subscripts, attr);
			else
				test_flag = banerjeeTest(write_subscripts, read_subscripts, attr);
			
			/* If we have detected that a dependency may exist or that the loop should be skipped, return immediately */
			if( (test_flag == 1) || (test_flag == 2) )
				return test_flag;
	
		}

	}

	/* If we get here, test_flag has remained 0, meaning that there are no dependencies */
	return test_flag;
}


/* Perform the simple ZIV test for array subscript expressions */
int ZIVTest(std::vector<SgExpression*> ref1, std::vector<SgExpression*> ref2)
{
	/* Check to make sure the expression lists are the same size (they should be) */
	if(ref1.size() != ref2.size())
	{
		std::cerr << "Array size mismatch in ZIVTest()" << std::endl;
		return 2;
	}
	
	/* Loop thru each array reference pair and check to see if there are no iter/symb var references in the pair */
	for(long unsigned int i = 0; i < ref1.size(); i++)
	{
		SgExpression *expr1 = ref1[i];
		SgExpression *expr2 = ref2[i];
		
		/* If any pair of array subcripts are two different INTs, there cannot possible be data dependence, so return 0 */
		if(isSgIntVal(expr1) && isSgIntVal(expr2))
			if( isSgIntVal(expr1)->get_value() != isSgIntVal(expr2)->get_value() )
				return 0;
	}

	/* If we get here, the test is inconclusive */
	return 1;
}


/* Perform the simple GCD dependency test for the array subscript expressions */
int GCDTest(std::vector<SgExpression*> ref1, std::vector<SgExpression*> ref2, LoopNestAttribute *attr)
{
	/* Check to make sure the expression lists are the same size (they should be) */
	if(ref1.size() != ref2.size())
	{
		std::cerr << "Array size mismatch in GCDTest()" << std::endl;
		return 2;
	}

	
	/* Obtain attributes */
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();
	std::list<std::string> loop_symb_vec = attr->get_symb_vec();
	int vec_size = loop_iter_vec.size() + loop_symb_vec.size();

	
	/* Perform the test for each of the subscripts */
	for(long unsigned int i = 0; i < ref1.size(); i++)
	{
		
		/*
		***********************************
	                Extract Coefficients       
		***********************************
		*/
		
		/* Convert the expressions into a constant-length vector and an int

                   Ex: [A*i+B*j+C] becomes
		   	[A,B] and C

		   Note: Symb vars are located at the end of the vector
		*/
		std::vector<int> coeff_vec1(vec_size), coeff_vec2(vec_size);
		int res1 = 0, res2 = 0;
		
		/* If coeff extraction fails, return 2 to skip the current loop (to be conservative) */
		if(!extractCoeff(ref1[i], coeff_vec1, res1, attr))
			return 2;
		if(!extractCoeff(ref2[i], coeff_vec2, res2, attr))
			return 2;

		/* Negate the second coeff_vec, append to the first coeff_vec, and subtract res2 from res1 */
		std::vector<int> final_coeff_vec = coeff_vec1;
		int final_res = res1 - res2;
		for(auto coeff_it = coeff_vec2.begin(); coeff_it != coeff_vec2.end(); coeff_it++)
			final_coeff_vec.push_back( (-1) * (*coeff_it) );

		//for(long unsigned int j = 0; j < final_coeff_vec.size(); j++)
		//	std::cout << "final_coeff_vec[" << j << "] = " << final_coeff_vec[j] << std::endl;
		//std::cout << "final_res = " << final_res << std::endl << std::endl;	
		
		/*
		**********************************
	               Perform the GCD test 
		**********************************
		*/
		
		/* Remove the zeros from the coeff vec */
		final_coeff_vec.erase(std::remove(final_coeff_vec.begin(), final_coeff_vec.end(), 0), final_coeff_vec.end());
		final_coeff_vec.shrink_to_fit();

		/* If the coeff vec was all zeros, then just continue */
		if(final_coeff_vec.size() == 0)
			continue;

		/* Get GCD */
		int gcd = final_coeff_vec[0];
		for(long unsigned int j = 1; j < final_coeff_vec.size(); j++)
		{
			gcd = euclidGCD(final_coeff_vec[j], gcd);

			if(gcd == 1)
				break;
		}
		
		/* If the gcd is 0, then all of the coeffs were 0, so our coeff_extraction could not get any useful info.  Therefore, go to next subscript */
		if(gcd == 0)
			continue;
		
		/* Check if gcd divides final_res evenly.  If not, return 0 (i.e. dependency cannot possibly exist) */
		if( (final_res % gcd) != 0 )
		       	return 0;	

		
	}



	/* If we get here, the test is inconclusive, so return 1 */
	return 1;
}


/* Extract the coefficients from the expressions */
bool extractCoeff(SgExpression *expr, std::vector<int> &coeff_vec, int &res, LoopNestAttribute *attr)
{
	/* Get iter and symb vecs */
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();
	std::list<std::string> loop_symb_vec = attr->get_symb_vec();

	/* Handle the following:
		A
	      	A*i		
		A*i + B			
		C*(A*i + B)	
	  	
		And any sum of those
	*/

	/* Case I: An integer constant */
	if(isSgValueExp(expr))
	{
		SgIntVal *int_val = isSgIntVal(expr);
		if(!int_val)
			return false;
		
		res += int_val->get_value();
		
		return true;	
	}
	
	/* Case II: Multiplication */
	else if(isSgMultiplyOp(expr))
	{
		SgExpression *lhs = isSgMultiplyOp(expr)->get_lhs_operand();
		SgExpression *rhs = isSgMultiplyOp(expr)->get_rhs_operand();

		if(isSgIntVal(lhs)) 
		{
			int coeff = isSgIntVal(lhs)->get_value();
			
			/* Handles A*i case */
			if(isSgVarRefExp(rhs))
			{
				/* Extract info */
				std::string var = isSgVarRefExp(rhs)->get_symbol()->get_name().getString();
			
				/* Find the iter var (if not successful, check for symb var) */
				std::list<std::string>::iterator var_it = std::find(loop_iter_vec.begin(), loop_iter_vec.end(), var);
				if(var_it == loop_iter_vec.end())
				{
					var_it = std::find(loop_symb_vec.begin(), loop_symb_vec.end(), var);
				
					/* If neither an iter_var or symb_var, we cannot reduce this expression so return false */
					if(var_it == loop_symb_vec.end())
						return false;

					/* If we get here, then the rhs was a symb_var and we append to proper spot */
					int symb_index = loop_iter_vec.size() + std::distance(loop_symb_vec.begin(), var_it);
					coeff_vec[symb_index] += coeff; 

				}
				else
				{
					/* If we get here, then rhs is an iter_var */
					int iter_index = std::distance(loop_iter_vec.begin(), var_it);
					coeff_vec[iter_index] += coeff;
				}
			}
			/* Handles C*(A*i +/- B) case */
			else if(isSgAddOp(rhs) || isSgSubtractOp(rhs))
			{
								
				/* Create temporary vector and result value */
				std::vector<int> temp_coeff_vec(coeff_vec.size());
				int temp_res = 0;

				/* Use recursion to get coeffs within inner expr */
				if(!extractCoeff(rhs, temp_coeff_vec, temp_res, attr))
					return false;

				/* Add the new coeffs to existing coeffs, with the appropriate multiplication */
				for(long unsigned int i = 0; i < coeff_vec.size(); i++)
						coeff_vec[i] += coeff*temp_coeff_vec[i];
				
				/* Add/Subtract the res */
				if(isSgAddOp(rhs))
					res += coeff*temp_res;
				else
					res += (-1)*coeff*temp_res;

			}
			else
				return false;

		}
	}

	/* Case III: Addition (or Subtraction) */
	else if( isSgAddOp(expr) || isSgSubtractOp(expr) )
	{
		//SgExpression *lhs = isSgAddOp(expr)->get_lhs_operand();
		//SgExpression *rhs = isSgAddOp(expr)->get_rhs_operand();
		SgExpression *lhs = isSgBinaryOp(expr)->get_lhs_operand();
		SgExpression *rhs = isSgBinaryOp(expr)->get_rhs_operand();
	
		/* Create temporary vectors and result values */
		std::vector<int> temp_coeff_vec_lhs(coeff_vec.size()), temp_coeff_vec_rhs(coeff_vec.size());
		int temp_res_lhs = 0, temp_res_rhs = 0;

		/* Use recursion to get coeffs within inner expr */
		if(!extractCoeff(lhs, temp_coeff_vec_lhs, temp_res_lhs, attr))
			return false;

		if(!extractCoeff(rhs, temp_coeff_vec_rhs, temp_res_rhs, attr))
			return false;

		/* Add (or subtract) the new coeffs to existing coeffs */
		if(isSgAddOp(expr))
		{
			for(long unsigned int i = 0; i < coeff_vec.size(); i++)
				coeff_vec[i] += temp_coeff_vec_lhs[i] + temp_coeff_vec_rhs[i];
			
			res += temp_res_lhs + temp_res_rhs;
		}

		else
		{
			for(long unsigned int i = 0; i < coeff_vec.size(); i++)
				coeff_vec[i] += temp_coeff_vec_lhs[i] - temp_coeff_vec_rhs[i];
			
			res += temp_res_lhs - temp_res_rhs;
		}
	}
	
	/* Case IV: Iter Var or Symbolic Constant (i.e. 1*i, but not expressed as a multiplication) */
	else if(isSgVarRefExp(expr))
	{
		/* Extract info */
		std::string var = isSgVarRefExp(expr)->get_symbol()->get_name().getString();
			
		/* Find the iter var (if not successful, check for symb var) */
		std::list<std::string>::iterator var_it = std::find(loop_iter_vec.begin(), loop_iter_vec.end(), var);
		if(var_it == loop_iter_vec.end())
		{
			var_it = std::find(loop_symb_vec.begin(), loop_symb_vec.end(), var);
				
			/* If neither an iter_var or symb_var, we cannot reduce this expression so return false */
			if(var_it == loop_symb_vec.end())
				return false;

			/* If we get here, then the expr is a symb_var and we append to proper spot */
			int symb_index = loop_iter_vec.size() + std::distance(loop_symb_vec.begin(), var_it);
			coeff_vec[symb_index] += 1; 

		}
		else
		{
			/* If we get here, then the expr is an iter_var */
			int iter_index = std::distance(loop_iter_vec.begin(), var_it);
			coeff_vec[iter_index] += 1;
		}

	}
	else
		return false;

	
	/* If we get here, then the expression was reducible and we updated the info in coeff_vec and res */
	return true;	
}


/* Use the Euclidean Algorithm to find the GCD of two INTs */
int euclidGCD(int a, int b)
{
	if(a == 0)
		return b;
	
	return euclidGCD(b % a, a);
}


/* Perform the Banerjee Test for dependency check */
int banerjeeTest(std::vector<SgExpression*> ref1, std::vector<SgExpression*> ref2, LoopNestAttribute *attr)
{
	/* Check to make sure the expression lists are the same size (they should be) */
	if(ref1.size() != ref2.size())
	{
		std::cerr << "Array size mismatch in GCDTest()" << std::endl;
		return 2;
	}

	
	/* Obtain attributes */
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();
	std::list<std::string> loop_symb_vec = attr->get_symb_vec();
	int vec_size = loop_iter_vec.size() + loop_symb_vec.size();
	
	/* Extract coefficients from bound expressions (slightly different from extractCoeff() function) */
	std::list<SgExpression*> loop_bound_vec = attr->get_bound_vec();
	std::vector<int> bound_vec;
	for(auto b_it = loop_bound_vec.begin(); b_it != loop_bound_vec.end(); b_it++)
	{
		SgExpression *bound = *b_it;

		/* Append INT values to the bound_vec */
		if(isSgIntVal(bound))
			bound_vec.push_back(isSgIntVal(bound)->get_value());

		/* Otherwise, check to see if there is a reference to a symb_val */
		else
		{	
			Rose_STL_Container<SgNode*> var_refs = NodeQuery::querySubTree(bound, V_SgVarRefExp);
			for(auto v_it = var_refs.begin(); v_it != var_refs.end(); v_it++)
			{
				std::string var = isSgVarRefExp(*v_it)->get_symbol()->get_name().getString();
				
				/* If we find a reference to it, make most conservative assumption for the bound (i.e. INT_MAX) */
				if( std::find(loop_symb_vec.begin(), loop_symb_vec.end(), var) != loop_symb_vec.end() )
				{
					bound_vec.push_back(INT_MAX);
					break;
				}
			}	
		}
	}

	/* If the sizes do not match, then we could not extract enough info from loop_bound_vec so the test cannot be completed */
	if(bound_vec.size() != loop_bound_vec.size())
		return 2;

	/* Perform the test for each of the subscripts */
	for(long unsigned int i = 0; i < ref1.size(); i++)
	{
		
		/*
		***********************************
	                Extract Coefficients       
		***********************************
		*/
		
		/* Convert the expressions into a constant-length vector and an int

                   Ex: [A*i+B*j+C] becomes
		   	[A,B] and C

		   Note: Symb vars are located at the end of the vector
		*/
		std::vector<int> a(vec_size), b(vec_size);
		int a0 = 0, b0 = 0;
		
		/* If coeff extraction fails, return 2 to skip the current loop (to be conservative) */
		if(!extractCoeff(ref1[i], a, a0, attr))
			return 2;
		if(!extractCoeff(ref2[i], b, b0, attr))
			return 2;

		/* diff has the value b0 - a0, which we use in the Banerjee inequality */
		int diff = b0 - a0;


		/*
		***************************************
	               Perform the Banerjee Test 
		***************************************
		*/

		/* There are three cases to deal with (corresponding to direction vectors: (<, =, >) )
		   
		   Recall:
		       a holds the coeffs of the first reference
		       b holds the coeffs of the second reference
		       diff holds the value of the constant (i.e. not a coeff of an iter or symb var)
		       bound_vec holds the upper bounds of the loop nest
		       The lower bound of each loop is 1, due to the normalization steps performed earlier in the analysis
		
		   A solution may exist if:   total_LB <= diff <= total_UB
		   A solution definitely does NOT exists if diff is not bounded by total_LB and total_UB for all three cases
		*/

		/* Case I: Direction Vector is < */
		int total_LB = 0, total_UB = 0;
		for(long unsigned int idx = 0; idx < a.size(); idx++)
		{
			total_LB += (-1)*pos((neg(a[idx]) + b[idx]))*(bound_vec[idx] - 1)  +  (neg(neg(a[idx]) + b[idx]) + pos(a[idx]))  -  b[idx];
	       	       	total_UB += pos(pos(a[idx]) - b[idx])*(bound_vec[idx] - 1)  -  (neg(pos(a[idx]) - b[idx]) + neg(a[idx]))  -  b[idx];
		}

		/* If diff is bounded by total_LB and total_UB, then a solution may exist, so continue onto the next pair */
		if( (total_LB <= diff) && (diff <= total_UB) )
			continue;
			

		/* Case II: Direction Vector is = */
		total_LB = 0;
		total_UB = 0;
		for(long unsigned int idx = 0; idx < a.size(); idx++)
		{
			total_LB += (-1)*neg(a[idx] - b[idx])*bound_vec[idx]  +  pos(a[idx] - b[idx]);
			total_UB += pos(a[idx] - b[idx])*bound_vec[idx]  -  neg(a[idx] - b[idx]);
		}

		/* Same check for boundedness */
		if( (total_LB <= diff) && (diff <= total_UB) )
			continue;


		/* Case III: Direction Vector is > */
		total_LB = 0;
		total_UB = 0;
		for(long unsigned int idx = 0; idx < a.size(); idx++)
		{
			total_LB += (-1)*neg(a[idx] - pos(b[idx]))*(bound_vec[idx] - 1)  +  (pos(a[idx] - pos(b[idx])) + neg(b[idx]))  +  a[idx];
			total_UB += pos(a[idx] + neg(b[idx]))*(bound_vec[idx] - 1)  -  (neg(a[idx] + neg(b[idx])) + pos(b[idx]))  +  a[idx]; 
		}

		/* Check for boundedness */
		if( (total_LB <= diff) && (diff <= total_UB) )
			continue;

		
		/* If we get here, diff was not bounded, so there does not exist a solution */
		return 0;
		
	}



	/* If we get here, the test is inconclusive, so return 1 */
	return 1;

}

/* Helper function to get positive part of a number */
int pos(int a)
{
	if(a >= 0)
		return a;
	
	return 0;
}


/* Helper function to get negative part of a number */
int neg(int a)
{
	if(a >= 0)
	       return 0;

	return (-1)*a;
}	
