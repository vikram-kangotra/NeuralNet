CXX = g++
CXXFLAGS = 
LDFLAGS = `pkg-config --cflags --libs opencv4`

APPNAME = deep

OBJDIR = obj
DEPDIR = dep

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:%.cpp=$(OBJDIR)/%.o)
DEPS = $(SRCS:%.cpp=$(DEPDIR)/%.d)

INCLUDE = 

all: $(APPNAME)

$(APPNAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(DEPDIR)/%.d: %.cpp
	@$(CXX) $(CFLAGS) $< -MM -MT $(@:$(DEPDIR)/%.d=$(OBJDIR)/%.o) >$@

-include $(DEPS)

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(INCLUDE) $(LDFLAGS)

clean:
	rm $(OBJS) $(DEPS) $(APPNAME)