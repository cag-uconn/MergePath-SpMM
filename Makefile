
CC = nvcc
CFLAGS  = -O3 

# The build target 
TARGET = row_wise

all: $(TARGET)

$(TARGET): src/$(TARGET).cu src/main.cpp
	$(CC) $(CFLAGS) -o $(TARGET) src/$(TARGET).cu src/main.cpp

clean:
	$(RM) $(TARGET)