
USER_DIR = .

CPPFLAGS += -isystem $(GTEST_DIR)/include

CXXFLAGS += -g -Wall -Wextra -pthread -std=c++11


TESTS = test nntest

GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

all : $(TESTS)


clean :
	rm -f $(TESTS) gtest.a gtest_main.a *.o  

GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

gradient.o : $(USER_DIR)/gradient.cc $(USER_DIR)/gradient.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/gradient.cc

predict.o : $(USER_DIR)/predict.cc $(USER_DIR)/predict.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/predict.cc

nn.o : $(USER_DIR)/nn.cc $(USER_DIR)/nn.hpp $(USER_DIR)/layer.hpp $(USER_DIR)/graph.hpp $(USER_DIR)/optimizer.hpp $(USER_DIR)/lossfunc.hpp $(USER_DIR)/gradient.hpp $(USER_DIR)/predict.hpp 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/nn.cc 

graph.o : $(USER_DIR)/graph.cc $(USER_DIR)/layer.hpp $(USER_DIR)/graph.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/graph.cc


layer.o : $(USER_DIR)/layer.cc $(USER_DIR)/layer.hpp $(USER_DIR)/input.hpp $(USER_DIR)/fully_connect.hpp $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/layer.cc

test_layer.o : $(USER_DIR)/test_layer.cc \
                     $(USER_DIR)/layer.hpp $(USER_DIR)/input.hpp $(USER_DIR)/fully_connect.hpp $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/test_layer.cc

test_nn.o : $(USER_DIR)/test_nn.cc $(USER_DIR)/nn.hpp $(USER_DIR)/layer.hpp $(USER_DIR)/graph.hpp  $(GTEST_HEADERS) $(USER_DIR)/gradient.hpp $(USER_DIR)/gradient.cc $(USER_DIR)/predict.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/test_nn.cc $(USER_DIR)/gradient.cc $(USER_DIR)/predict.cc

test : test_layer.o layer.o gtest_main.a gradient.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

nntest : test_nn.o nn.o graph.o gtest_main.a layer.o gradient.o predict.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@


astyle : 
	astyle --style=kr --indent=spaces=4 --indent-switches --suffix=none *.cc
	astyle --style=kr --indent=spaces=4 --indent-switches --suffix=none *.hpp

