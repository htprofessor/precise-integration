CXX = g++
CXXFLAGS = -I/intel/oneapi/mkl/2024.2/include -O2 -fopenmp
LDFLAGS = -L/rshome/yingchen.zhao/intel/oneapi/mkl/2024.2/lib -lmkl_rt -lpthread -lm -ldl -fopenmp

TARGET = time_invarying 

SRCS = main.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
