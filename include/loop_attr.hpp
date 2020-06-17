/* Header File for the LoopNestAttribute Class */

#ifndef LOOP_ATTR
#define LOOP_ATTR

#include "rose.h"

/* Class for setting attributes of loop nest */
class LoopNestAttribute : public AstAttribute {
	public:
		LoopNestAttribute(int s, bool f) {this->size = s; this->flag = f;}
		virtual LoopNestAttribute * copy() const override {return new LoopNestAttribute(*this);}
		virtual std::string attribute_class_name() const override {return "LoopNestAttribute";}
		int get_nest_size() {return size;}
		bool get_nest_flag() {return flag;}
		std::list<std::string> get_iter_vec() {return iter_vec;}
		void set_nest_flag(bool new_flag) {flag = new_flag;}
		void set_iter_vec(std::list<std::string> vec) {iter_vec = vec;}
	private:
		int size;
		bool flag;
		std::list<std::string> iter_vec;
};
#endif
