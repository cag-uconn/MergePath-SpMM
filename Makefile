
CC = nvcc
CFLAGS  = -O3 

# The build target 
ROW_WISE = row_wise
NZ_SPLITTING = nz_splitting

all: $(ROW_WISE) $(NZ_SPLITTING)

$(ROW_WISE): src/$(ROW_WISE).cu 
	$(CC) $(CFLAGS) -o $(ROW_WISE) src/$(ROW_WISE).cu

$(NZ_SPLITTING): src/$(NZ_SPLITTING).cu 
	$(CC) $(CFLAGS) -o $(NZ_SPLITTING) src/$(NZ_SPLITTING).cu

clean:
	$(RM) $(ROW_WISE) $(NZ_SPLITTING)