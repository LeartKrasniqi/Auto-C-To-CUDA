/* Implementation of preprocessinf functions */
#include "./preprocess.hpp"

/* Attempts to convert a while-loop nest into a for-loop nest */
SgStatement * convertWhileToFor(SgWhileStmt *loop_nest)
{
	/* Create the bb which will represent the new for-loop nest */
	SgBasicBlock *bb_nest = SageBuilder::buildBasicBlock();
	bb_nest->set_parent(loop_nest->get_parent());

	/* Go through each while loop in the loop nest */
	Rose_STL_Container<SgNode*> while_loops = NodeQuery::querySubTree(loop_nest, V_SgWhileStmt);
	for(size_t i = 0; i < while_loops.size(); i++)
	{
		SgWhileStmt *loop = isSgWhileStmt(while_loops[i]);

		/* First, get the index variable */
		SgVarRefExp *index_var;
		SgStatement *cond_stmt = loop->get_condition();

		/* Check if condition is an SgVarRefExp, or a binop */
		SgExprStatement *cond_expr_stmt = isSgExprStatement(cond_stmt);
		if(!cond_expr_stmt)
			return NULL;

		SgExpression *cond_expr = cond_expr_stmt->get_expression();
		if(isSgVarRefExp(cond_expr))
			index_var = isSgVarRefExp(cond_expr);
		else if(isSgBinaryOp(cond_expr))
		{
			if(isSgVarRefExp(isSgBinaryOp(cond_expr)->get_lhs_operand()))
				index_var = isSgVarRefExp(isSgBinaryOp(cond_expr)->get_lhs_operand());
			else 
				return NULL;
		}
		else
			return NULL;

		/* Check if previous statement is an initialization of the index variable -- Keep check within scope */
		SgExprStatement *init_stmt = isSgExprStatement(SageInterface::getPreviousStatement(loop, false));
		if(!init_stmt)
			return NULL;

		SgBinaryOp *prev_expr = isSgBinaryOp(init_stmt->get_expression());
		if(!prev_expr)
			return NULL;

		if(isSgVarRefExp(prev_expr->get_lhs_operand()))
		{
			if( isSgVarRefExp(prev_expr->get_lhs_operand())->get_symbol()->get_declaration() != index_var->get_symbol()->get_declaration() )
				return NULL;
		}
		else
			return NULL;


		/* If we get here, the previous statement was an initialization of the index variable */
		

		/* Now, make sure that there are no writes to the index variable or the bounds variable within the loop body */
		SgBasicBlock *body = isSgBasicBlock(loop->get_body());

		/* Create a set that will hold the index var and the bounds vars */
		std::set<SgInitializedName*> idx_bound_vars;
		idx_bound_vars.insert(index_var->get_symbol()->get_declaration());
		if(isSgBinaryOp(cond_expr))
		{
			Rose_STL_Container<SgNode*> v = NodeQuery::querySubTree(isSgBinaryOp(cond_expr)->get_rhs_operand(), V_SgVarRefExp);
			for(auto it = v.begin(); it != v.end(); it++)
				idx_bound_vars.insert(isSgVarRefExp(*it)->get_symbol()->get_declaration());
		}

		/* Go through every statement in the body of the loop (except the last one) and check if there are any writes to the index/bound vars */
		SgStatementPtrList &body_stmts = body->get_statements();
		for(size_t j = 0; j < body_stmts.size() - 1; j++)
		{
			/* Get the reads/writes in this statement */
			std::set<SgInitializedName*> reads, writes;
			SageInterface::collectReadWriteVariables(body_stmts[j], reads, writes);

			/* Check if any of the writes are to the index/bound vars.  If so, return false. */
			for(auto w_it = writes.begin(); w_it != writes.end(); w_it++)
				if( std::find(idx_bound_vars.begin(), idx_bound_vars.end(), *w_it) != idx_bound_vars.end() )
					return NULL;
		}

		/* Check if the last statement is an update to the index variable */
		SgExprStatement *stride_stmt = isSgExprStatement(body_stmts.back());
		if(!stride_stmt)
			return NULL;
		std::set<SgInitializedName*> reads, writes;
		SageInterface::collectReadWriteVariables(stride_stmt, reads, writes);
		if( std::find(writes.begin(), writes.end(), index_var->get_symbol()->get_declaration()) == writes.end() )
			return NULL;

		
		/* If we get here, then the loop passes our structure, so we can convert it into a for loop */		
		SgBasicBlock *for_body = SageBuilder::buildBasicBlock();
		for(size_t j = 0; j < body_stmts.size() - 1; j++)
			SageInterface::appendStatement(body_stmts[j], for_body);
		
		/* If the cond_stmt is just a var ref, convert it into a > 0 binop */
		SgStatement *test_stmt = cond_stmt;
		if(isSgVarRefExp(cond_expr))
			test_stmt = SageBuilder::buildExprStatement(
								SageBuilder::buildGreaterThanOp(
											index_var,
											SageBuilder::buildIntVal(0) ) );



		SgForStatement *for_loop = SageBuilder::buildForStatement(init_stmt, test_stmt, stride_stmt->get_expression(), for_body);
		for_loop->set_parent(bb_nest);

		/* Append only the first while loop to bb_nest, as the rest will be taken care of inside of the first loop */
		if(i == 0)
			SageInterface::appendStatement(for_loop, bb_nest);
		else
			isSgStatement(loop->get_parent())->replace_statement(loop, for_loop);

	}
	
	/* If we get here, all of the loops have been successfully converted.  So, we return bb_nest */
	return bb_nest;
}

