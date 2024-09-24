CXX = g++
CXXFLAGS = -I/path/to/mkl/include -O2 -fopenmp
LDFLAGS = -L/path/to/mkl/lib -lmkl_rt -lpthread -lm -ldl -fopenmp

TARGET = riccati_pim_solver

SRCS = riccati_pim.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
