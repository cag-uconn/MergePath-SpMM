
CC = nvcc
CFLAGS  = -O3 

# The build target 
ROW_WISE = row_wise
NZ_SPLITTING = nz_splitting
MP = MergePathSpMM
CUSPARSE_ROW = cusparse_spmm_row
CUSPARSE_COL = cusparse_spmm_col

all: $(ROW_WISE) $(NZ_SPLITTING) $(MP) $(CUSPARSE_ROW) $(CUSPARSE_COL)

$(ROW_WISE): src/$(ROW_WISE).cu 
	$(CC) $(CFLAGS) -o src/$(ROW_WISE) src/$(ROW_WISE).cu

$(NZ_SPLITTING): src/$(NZ_SPLITTING).cu 
	$(CC) $(CFLAGS) -o src/$(NZ_SPLITTING) src/$(NZ_SPLITTING).cu
	
$(MP): src/$(MP).cu
	$(CC) $(CFLAGS) -o src/$(MP) src/$(MP).cu
	
$(CUSPARSE_ROW): src/$(CUSPARSE_ROW).cu
	$(CC) $(CFLAGS) -lcusparse -o src/$(CUSPARSE_ROW) src/$(CUSPARSE_ROW).cu

$(CUSPARSE_COL): src/$(CUSPARSE_COL).cu
	$(CC) $(CFLAGS) -lcusparse -o src/$(CUSPARSE_COL) src/$(CUSPARSE_COL).cu

clean:
	$(RM) src/$(ROW_WISE) src/$(NZ_SPLITTING) src/$(MP) src/$(CUSPARSE_ROW) src/$(CUSPARSE_COL)
