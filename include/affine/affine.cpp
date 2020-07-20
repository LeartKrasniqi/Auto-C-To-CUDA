/* Implementation of affine test functions */
#include "affine.hpp"
#include <algorithm>

/* Test whether loop nest is affine */
bool affineTest(SgForStatement *loop_nest)
{
	/* Obtain the attributes of the loop */
	LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();
	std::list<SgExpression*> loop_bound_vec = attr->get_bound_vec();
	std::list<std::string> loop_symb_vec = attr->get_symb_vec();
	int loop_nest_size = attr->get_nest_size();	
	
	/* Check to see if there are any function calls in the loop. If so, return false to remain conservative */
	Rose_STL_Container<SgNode*> fn_calls = NodeQuery::querySubTree(loop_nest, V_SgFunctionCallExp);
	if(fn_calls.size() > 0)
		return false;
	
	/* Obtain the body of the loop nest (assuming the nest is perfectly nested) */
	Rose_STL_Container<SgNode*> inner_loops = NodeQuery::querySubTree(loop_nest, V_SgForStatement);
	SgStatement *body = isSgForStatement(inner_loops[loop_nest_size - 1])->get_loop_body();
	
	/* Make sure there are no writes to any of the iter variables (since SageInterface::collectReadWriteVariables() cannot handle these cases due to our normalization) */
	SgStatementPtrList &stmt_list = isSgBasicBlock(body)->get_statements();
	for(auto it = stmt_list.begin(); it != stmt_list.end(); it++)
	{
		if(!isSgExprStatement(*it))
			continue;
		
		/* Obtain the variable references of LHS of binary op, or operand in a unary op */
		SgExpression *expr = isSgExprStatement(*it)->get_expression();
		Rose_STL_Container<SgNode*> var_refs;
		if( isSgBinaryOp(expr) && !isSgPntrArrRefExp(isSgBinaryOp(expr)->get_lhs_operand()) )
			var_refs = NodeQuery::querySubTree(isSgBinaryOp(expr)->get_lhs_operand(), V_SgVarRefExp);
		else if(isSgUnaryOp(expr))
			var_refs = NodeQuery::querySubTree(isSgUnaryOp(expr)->get_operand(), V_SgVarRefExp);
		else
			continue;

		/* Check to see if these references are to any of the iter vals */
		for(Rose_STL_Container<SgNode*>::iterator v_it = var_refs.begin(); v_it != var_refs.end(); v_it++)
		{
			std::string var = isSgVarRefExp(*v_it)->get_symbol()->get_name().getString();
			if( std::find(loop_iter_vec.begin(), loop_iter_vec.end(), var) != loop_iter_vec.end() )
				return false;
		}
	}

				
	/* Check to make sure that the values in loop_symb_vec are actually constant (i.e. loop invariant) */
	std::set<SgInitializedName*> read_vars, write_vars;
	SageInterface::collectReadWriteVariables(body, read_vars, write_vars); 
	for(std::set<SgInitializedName*>::iterator w_it = write_vars.begin(); w_it != write_vars.end(); w_it++)	
		if( std::find(loop_symb_vec.begin(), loop_symb_vec.end(), (*w_it)->get_name().getString()) != loop_symb_vec.end() )
		      	return false;	

	/* Check if loop bounds are affine */
	for(std::list<SgExpression*>::iterator bound_iter = loop_bound_vec.begin(); bound_iter != loop_bound_vec.end(); bound_iter++)
		if(!affineTestExpr(*bound_iter, loop_nest))
			return false;
	
	/* Get the array references */
	Rose_STL_Container<SgNode*> array_refs = NodeQuery::querySubTree(body, V_SgPntrArrRefExp); 
	
	/* Loop through each reference and check if affine */
	Rose_STL_Container<SgNode*>::iterator arr_iter;
	for(arr_iter = array_refs.begin(); arr_iter != array_refs.end(); arr_iter++)
	{
		SgExpression *expr = isSgPntrArrRefExp(*arr_iter)->get_rhs_operand();
		if(expr)
		{
			/* Perform any constant folding */
			SgExpression *new_expr = indexConstantFolding(expr);
			
			/* Set the new array index expression */
			isSgPntrArrRefExp(*arr_iter)->set_rhs_operand(new_expr);
			
			/* Perform built-in constant folding now */
			SageInterface::constantFolding(*arr_iter);
		
			/* Check whether expression is affine */
			if(!affineTestExpr(new_expr, loop_nest))
				return false;
		}
	}



	/* If we get here, all expressions within the body are affine, so the loop nest is affine */
	return true;
}


/* Perform some additional constant folding in the array expression */
SgExpression * indexConstantFolding(SgExpression *expr)
{	
	SgBinaryOp *op = isSgBinaryOp(expr);
	
	if(!op)
		return expr;
	
	/* Four cases we deal with:
	     A*i + B		--> No possible folding
	     A*i + B +/- C  	--> A*i + (B +/- C)
	     D*(A*i + B)	--> (D*A)*i + (D*B)
	     D*(A*i + B) +/- C  --> (D*A)*i + (D*B +/- C)
	*/

	/* Will be the new expression */
	SgExpression *new_expr;
	
	/* Cases I and IV */
	if( (isSgAddOp(op) || isSgSubtractOp(op)) && isSgMultiplyOp(op->get_lhs_operand()) )
	{
		/* If mult_op->rhs() is not an add_op, we are in case I, so just return */
		if(!isSgAddOp(isSgBinaryOp(op->get_lhs_operand())->get_rhs_operand()))
			return expr;

		/* Obtain A, i, B, C, and D */
	 	SgExpression *A, *i, *B, *C, *D;	
		
		/* Keeping these just in case since I needed to add error checking and it looks less fancy now :( */
		/*
		SgExpression *A = isSgBinaryOp(
				  isSgBinaryOp(
				  isSgBinaryOp(
					op->get_lhs_operand())
				        ->get_rhs_operand())
				        ->get_lhs_operand())
			                ->get_lhs_operand();
		SgExpression *i = isSgBinaryOp(
				  isSgBinaryOp(
				  isSgBinaryOp(	
					op->get_lhs_operand())
				  	->get_rhs_operand())
					->get_lhs_operand())
					->get_rhs_operand();
		SgExpression *B = isSgBinaryOp(
				  isSgBinaryOp(
					op->get_lhs_operand())
					->get_rhs_operand())
					->get_rhs_operand();
		*/
		
		SgBinaryOp *A_intermed = isSgBinaryOp(isSgBinaryOp(op->get_lhs_operand())->get_rhs_operand());
		if(A_intermed)
			if( (A_intermed = isSgBinaryOp(A_intermed->get_lhs_operand())) )
				A = A_intermed->get_lhs_operand();
			else
				return expr;
		else
			return expr;


		SgBinaryOp *i_intermed = isSgBinaryOp(isSgBinaryOp(op->get_lhs_operand())->get_rhs_operand());
		if(i_intermed)
			if( (i_intermed = isSgBinaryOp(i_intermed->get_lhs_operand())) )
				i = i_intermed->get_rhs_operand();
			else
				return expr;
		else
			return expr;

	
		SgBinaryOp *B_intermed = isSgBinaryOp(isSgBinaryOp(op->get_lhs_operand())->get_rhs_operand());
		if(B_intermed)
			B = B_intermed->get_rhs_operand();
		else
			return expr;

		
		C = op->get_rhs_operand();
		D = isSgBinaryOp(op->get_lhs_operand())->get_lhs_operand();

		/* Perform distribution */
		SgExpression *slope = SageBuilder::buildMultiplyOp( SageBuilder::buildMultiplyOp(D, A) , i);
		SgExpression *intermed = SageBuilder::buildMultiplyOp(D, B);

		SgExpression *y_inter;
		if(isSgAddOp(op))
			y_inter = SageBuilder::buildAddOp(intermed, C);
		else
			y_inter = SageBuilder::buildSubtractOp(intermed, C);

		/* Create the new expression */
		new_expr = SageBuilder::buildAddOp(slope, y_inter);
 
	}
	/* Case II */
	else if( (isSgAddOp(op) || isSgSubtractOp(op)) && isSgAddOp(op->get_lhs_operand()) )
	{
		/* This should be A*i */
		SgExpression *inner_lhs = isSgBinaryOp(op->get_lhs_operand())->get_lhs_operand();

		/* This should be B */
		SgExpression *inner_rhs = isSgBinaryOp(op->get_lhs_operand())->get_rhs_operand();

		/* This should be C */
		SgExpression *outer_rhs = op->get_rhs_operand();

		/* Holds intermediate calculation */
		SgExpression *intermed;
		if(isSgAddOp(op))
			intermed = SageBuilder::buildAddOp(inner_rhs, outer_rhs);
		else
			intermed = SageBuilder::buildSubtractOp(inner_rhs, outer_rhs);

		/* Create the new expression */
		new_expr = SageBuilder::buildAddOp(inner_lhs, intermed);

	}
	/* Case III */
	else if(isSgMultiplyOp(op))
	{	
		/* This should be (A*i+B) */
		SgBinaryOp *inner_rhs = isSgBinaryOp(op->get_rhs_operand());
		if(!inner_rhs)
			return expr;

		/* Obtain A, i, B, and D */
		SgExpression *A, *i, *B, *D;
		
		if(isSgBinaryOp(inner_rhs->get_lhs_operand()))
			A = isSgBinaryOp(inner_rhs->get_lhs_operand())->get_lhs_operand();
		else
			return expr;

		if(isSgBinaryOp(inner_rhs->get_lhs_operand()))
			i = isSgBinaryOp(inner_rhs->get_lhs_operand())->get_rhs_operand();
		else
			return expr;

		B = inner_rhs->get_rhs_operand();
		D = op->get_lhs_operand();

		/* Distributive property time */
		SgExpression *slope = SageBuilder::buildMultiplyOp( SageBuilder::buildMultiplyOp(D, A) , i);
		SgExpression *y_inter = SageBuilder::buildMultiplyOp(D, B);
		
		/* Create the new expr */
		new_expr = SageBuilder::buildAddOp(slope, y_inter);
	
	}
	else
		return expr;

	
	return new_expr;
}


/* Perform affine check on single expression */
bool affineTestExpr(SgExpression *expr, SgForStatement *loop_nest)
{
	/* Expression is affine iff is it of form A*i + B*j + C*k + ... + Z, where (i,j,k,..) are iter_vars or symbolic consts */
	
	/* Obtain attributes of loop */
	LoopNestAttribute *attr = dynamic_cast<LoopNestAttribute*>(loop_nest->getAttribute("LoopNestInfo"));
	std::list<std::string> loop_iter_vec = attr->get_iter_vec();
	std::list<SgExpression*> loop_bound_vec = attr->get_bound_vec();
	std::list<std::string> loop_symb_vec = attr->get_symb_vec();	

	/* If exp is a constant, return true */
	if(isSgValueExp(expr))
		return true;

	if(isSgBinaryOp(expr))
	{
		SgExpression *lhs = isSgBinaryOp(expr)->get_lhs_operand();
		SgExpression *rhs = isSgBinaryOp(expr)->get_rhs_operand();
		
		/* An add/sub op is affine iff both lhs and rhs are affine */
		if( isSgAddOp(expr) || isSgSubtractOp(expr) )
			return ( affineTestExpr(lhs, loop_nest) && affineTestExpr(rhs, loop_nest) );
		
		/* A multiply op is affine iff it is A*i or i*A */
		if( isSgMultiplyOp(expr) )
		{
			if(isSgVarRefExp(lhs))
			{
				/* Right multiply by constant --> Affine */
				if(isSgValueExp(rhs))
					return true;

				/* Check to see if this is an iter_var or symb_var */
				bool isIterSymbVal = false;
				std::string lhs_string = isSgVarRefExp(lhs)->get_symbol()->get_name().getString();
				if( std::find(loop_iter_vec.begin(), loop_iter_vec.end(), lhs_string) != loop_iter_vec.end() )
					isIterSymbVal = true;
				else if( std::find(loop_symb_vec.begin(), loop_symb_vec.end(), lhs_string) != loop_symb_vec.end() )
					isIterSymbVal = true;


				/* Otherwise, check to make sure there is no reference to another iter_var or symb_var */
				Rose_STL_Container<SgNode*> v_ref = NodeQuery::querySubTree(rhs, V_SgVarRefExp);
				for(Rose_STL_Container<SgNode*>::iterator v_it = v_ref.begin(); v_it != v_ref.end(); v_it++)
				{
					std::string v_string = isSgVarRefExp(*v_it)->get_symbol()->get_name().getString();
					
					/* Check the iter_vec vals */
					if( std::find(loop_iter_vec.begin(), loop_iter_vec.end(), v_string) != loop_iter_vec.end() )
						if(isIterSymbVal)
							return false;

					/* Check the symb_vec vals */
					if( std::find(loop_symb_vec.begin(), loop_symb_vec.end(), v_string) != loop_symb_vec.end() )
						if(isIterSymbVal)
							return false;

				}
			
				/* If we get here, rhs and lhs dont both refer to iter or symb vars, so just check their affinities */
				return ( affineTestExpr(rhs, loop_nest) && lhs->get_type()->isIntegerType() );
	
			}

			if(isSgVarRefExp(rhs))
			{
				/* Left multiply by constant --> Affine */
				if(isSgValueExp(lhs))
					return true;

				/* Check to see if this is an iter_var or symb_var */
				bool isIterSymbVal = false;
				std::string rhs_string = isSgVarRefExp(rhs)->get_symbol()->get_name().getString();
				if( std::find(loop_iter_vec.begin(), loop_iter_vec.end(), rhs_string) != loop_iter_vec.end() )
					isIterSymbVal = true;
				else if( std::find(loop_symb_vec.begin(), loop_symb_vec.end(), rhs_string) != loop_symb_vec.end() )
					isIterSymbVal = true;


				/* Otherwise, check to make sure there is no reference to another iter_var or symb_var */
				Rose_STL_Container<SgNode*> v_ref = NodeQuery::querySubTree(lhs, V_SgVarRefExp);
				for(Rose_STL_Container<SgNode*>::iterator v_it = v_ref.begin(); v_it != v_ref.end(); v_it++)
				{
					std::string v_string = isSgVarRefExp(*v_it)->get_symbol()->get_name().getString();
					
					/* Check the iter_vec vals */
					if( std::find(loop_iter_vec.begin(), loop_iter_vec.end(), v_string) != loop_iter_vec.end() )
						if(isIterSymbVal)
							return false;

					/* Check the symb_vec vals */
					if( std::find(loop_symb_vec.begin(), loop_symb_vec.end(), v_string) != loop_symb_vec.end() )
						if(isIterSymbVal)
							return false;

				}
			
				/* If we get here, rhs and lhs dont both refer to iter or symb vars, so just check their affinities */
				return ( rhs->get_type()->isIntegerType() && affineTestExpr(lhs, loop_nest) );

			}

			/* Otherwise, check to see if both lhs and rhs are affine */
			return ( affineTestExpr(lhs, loop_nest) && affineTestExpr(rhs, loop_nest) );
		}

		/* Make sure rhs (i.e. divisor) is INT and lhs (i.e. dividend) is affine */
		if( isSgIntegerDivideOp(expr) || isSgDivideOp(expr) )
			return ( affineTestExpr(lhs, loop_nest) && isSgIntVal(rhs) );
	}

	/* If the expression is just a single variable, this is equivalent to 1*i, so it should be affine as long as i is of integer type */
	if(isSgVarRefExp(expr))
		return expr->get_type()->isIntegerType();

	
	/* If we get here, the expression did not match any of our accepted affine forms */
	return false;
}

