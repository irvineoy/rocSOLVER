template = f'''

I am currently working on GPU kernel optimization. Please help me write a benchmark command to test FUNCTION_NAME based on the following context.  I will use a GPU profiling tool along with this command to verify GPU performance. Please test single precision with slightly larger matrix dimensions, preferably around 4096. There is no need to verify results or perform performance analysis. Please provide a command that you deem appropriate as the final output. Your output should only include an executable rocsolver-bench command and nothing else.


rocsolver-bench context:
~/rocSOLVER# build/release/clients/staging/rocsolver-bench --help

rocSOLVER benchmark client help.

Usage: build/release/clients/staging/rocsolver-bench <options>

In addition to some common general options, the following list of options corresponds to all the parameters
that might be needed to test a given rocSOLVER function. The parameters are named as in the API user guide.
The arrays are initialized internally by the program with random values.

Note: When a required parameter/option is not provided, it will take the default value as listed below.
If no default value is defined, the program will try to calculate a suitable value depending on the context
of the problem and the tested function; if this is not possible, the program will abort with an error.
Functions that accept multiple size parameters can generally be provided a single size parameter (typically,
m) and a square-size matrix will be assumed.

Example: build/release/clients/staging/rocsolver-bench -f getf2_batched -m 30 --lda 75 --batch_count 350
This will test getf2_batched with a set of 350 random 30x30 matrices. strideP will be set to be equal to 30.

Options:
--help |-h                 Produces this help message. 

--batch_count <value>      Number of matrices or problem instances in the batch.
                           Only applicable to batch routines.
                             (Default value is: 1)

--device <value>           Set the default device to be used for subsequent program runs.
                             (Default value is: 0)

--function |-f <value>     The LAPACK function to test.
                           Options are: getf2, getrf, gesvd_batched, etc.
                             (Default value is: potf2)

--iters |-i <value>        Iterations to run inside the GPU timing loop.
                           Reported time will be the average.
                             (Default value is: 10)

--alg_mode <value>         0 = GPU-only, 1 = Hybrid
                           This will change how the algorithm operates.
                           Only applicable to functions with hybrid support.
                             (Default value is: 0)

--mem_query <value>        Calculate the required amount of device workspace memory? 0 = No, 1 = Yes.
                           This forces the client to print only the amount of device memory required by
                           the function, in bytes.
                             (Default value is: 0)

--perf <value>             Ignore CPU timing results? 0 = No, 1 = Yes.
                           This forces the client to print only the GPU time and the error if requested.
                             (Default value is: 0)

--precision |-r <value>    Precision to be used in the tests.
                           Options are: s, d, c, z.
                             (Default value is: s)

--profile <value>          Print profile logging results for the tested function.
                           The argument specifies the max depth of the nested output.
                           If the argument is unset or <= 0, profile logging is disabled.
                             (Default value is: 0)

--profile_kernels <value>  Include kernels in profile logging results? 0 = No, 1 = Yes.
                           Used in conjunction with --profile to include kernels in the profile log.
                             (Default value is: 0)

--singular <value>         Test with degenerate matrices? 0 = No, 1 = Yes
                           This will produce matrices that are singular, non positive-definite, etc.
                             (Default value is: 0)

--verify |-v <value>       Validate GPU results with CPU? 0 = No, 1 = Yes.
                           This will additionally print the relative error of the computations.
                             (Default value is: 0)

--hash <value>             Print hash of GPU results? 0 = No, 1 = Yes.
                           Meant for checking reproducibility of computations.
                             (Default value is: 0)

-k <value>                 Matrix/vector size parameter.
                           Represents a sub-dimension of a problem.
                           For example, the number of Householder reflections in a transformation.
                            

-m <value>                 Matrix/vector size parameter.
                           Typically, the number of rows of a matrix.
                            

-n <value>                 Matrix/vector size parameter.
                           Typically, the number of columns of a matrix,
                           or the order of a system or transformation.
                            

--nrhs <value>             Matrix/vector size parameter.
                           Typically, the number of columns of a matrix on the right-hand side of a problem.
                            

--inca <value>             Matrix/vector increment parameter.
                           Increment between values in matrices A.
                             (Default value is: 1)

--incb <value>             Matrix/vector increment parameter.
                           Increment between values in matrices B.
                             (Default value is: 1)

--incc <value>             Matrix/vector increment parameter.
                           Increment between values in matrices C.
                             (Default value is: 1)

--incx <value>             Matrix/vector increment parameter.
                           Increment between values in matrices/vectors X.
                             (Default value is: 1)

--lda <value>              Matrix size parameter.
                           Leading dimension of matrices A.
                            

--ldb <value>              Matrix size parameter.
                           Leading dimension of matrices B.
                            

--ldc <value>              Matrix size parameter.
                           Leading dimension of matrices C.
                            

--ldt <value>              Matrix size parameter.
                           Leading dimension of matrices T.
                            

--ldu <value>              Matrix size parameter.
                           Leading dimension of matrices U.
                            

--ldv <value>              Matrix size parameter.
                           Leading dimension of matrices V.
                            

--ldw <value>              Matrix size parameter.
                           Leading dimension of matrices W.
                            

--ldx <value>              Matrix size parameter.
                           Leading dimension of matrices X.
                            

--ldy <value>              Matrix size parameter.
                           Leading dimension of matrices Y.
                            

--ldz <value>              Matrix size parameter.
                           Leading dimension of matrices Z.
                            

--strideA <value>          Matrix/vector stride parameter.
                           Stride for matrices/vectors A.
                            

--strideB <value>          Matrix/vector stride parameter.
                           Stride for matrices/vectors B.
                            

--strideD <value>          Matrix/vector stride parameter.
                           Stride for matrices/vectors D.
                            

--strideE <value>          Matrix/vector stride parameter.
                           Stride for matrices/vectors E.
                            

--strideF <value>          Matrix/vector stride parameter.
                           Stride for vectors ifail.
                            

--strideQ <value>          Matrix/vector stride parameter.
                           Stride for vectors tauq.
                            

--strideP <value>          Matrix/vector stride parameter.
                           Stride for vectors tau, taup, and ipiv.
                            

--strideS <value>          Matrix/vector stride parameter.
                           Stride for matrices/vectors S.
                            

--strideU <value>          Matrix/vector stride parameter.
                           Stride for matrices/vectors U.
                            

--strideV <value>          Matrix/vector stride parameter.
                           Stride for matrices/vectors V.
                            

--strideW <value>          Matrix/vector stride parameter.
                           Stride for matrices/vectors W.
                            

--strideX <value>          Matrix/vector stride parameter.
                           Stride for matrices/vectors X.
                            

--strideZ <value>          Matrix/vector stride parameter.
                           Stride for matrices/vectors Z.
                            

--nnzM <value>             The number of non-zero elements in sparse matrix M.
                           Currently only a few test cases can be generated.
                           The benchmark client will use the available case closest to the input value.
                            

--nnzA <value>             The number of non-zero elements in sparse matrix A.
                           Currently only a few test cases can be generated.
                           The benchmark client will use the available case closest to the input value.
                            

--nnzL <value>             The number of non-zero elements in sparse matrix L.
                           Currently only a few test cases can be generated.
                           The benchmark client will use the available case closest to the input value.
                            

--nnzU <value>             The number of non-zero elements in sparse matrix U.
                           Currently only a few test cases can be generated.
                           The benchmark client will use the available case closest to the input value.
                            

--nnzT <value>             The number of non-zero elements in sparse matrix T.
                           Currently only a few test cases can be generated.
                           The benchmark client will use the available case closest to the input value.
                            

--rfinfo_mode <value>      Specifies the desired re-factorization algorithm.
                           1 = LU, 2 = Cholesky.
                            

--nc <value>               The number of columns of matrix C.
                           Only applicable to bdsqr.
                             (Default value is: 0)

--nu <value>               The number of columns of matrix U.
                           Only applicable to bdsqr.
                            

--nv <value>               The number of columns of matrix V.
                           Only applicable to bdsqr.
                             (Default value is: 0)

--svect <value>            N = none, S or V = the singular vectors are computed.
                           Indicates how the left singular vectors are to be calculated and stored.
                           Only applicable to bdsvdx.
                             (Default value is: N)

--k1 <value>               First index for row interchange.
                           Only applicable to laswp.
                            

--k2 <value>               Last index for row interchange.
                           Only applicable to laswp.
                            

--left_svect <value>       N = none, A = the entire orthogonal matrix is computed,
                           S or V = the singular vectors are computed,
                           O = the singular vectors overwrite the original matrix.
                           Indicates how the left singular vectors are to be calculated and stored.
                             (Default value is: N)

--right_svect <value>      N = none, A = the entire orthogonal matrix is computed,
                           S or V = the singular vectors are computed,
                           O = the singular vectors overwrite the original matrix.
                           Indicates how the right singular vectors are to be calculated and stored.
                             (Default value is: N)

--nev <value>              Number of eigenvectors to compute in a partial decomposition.
                           Only applicable to stein.
                            

--diag <value>             N = non-unit triangular, U = unit triangular.
                           Indicates whether the diagonal elements of a triangular matrix are assumed to be one.
                           Only applicable to trtri.
                             (Default value is: N)

--eorder <value>           E = entire matrix, B = by blocks.
                           Indicates whether the computed eigenvalues are ordered by blocks or for the entire matrix.
                           Only applicable to stebz.
                             (Default value is: E)

--nb <value>               Number of rows and columns in each block.
                           Only applicable to block tridiagonal matrix APIs.
                            

--nblocks <value>          Number of blocks along the diagonal.
                           Only applicable to block tridiagonal matrix APIs.
                            

--il <value>               Lower index in ordered subset of eigenvalues.
                           Used in partial eigenvalue decomposition functions.
                            

--iu <value>               Upper index in ordered subset of eigenvalues.
                           Used in partial eigenvalue decomposition functions.
                            

--erange <value>           A = all eigenvalues, V = in (vl, vu], I = from the il-th to the iu-th.
                           For partial eigenvalue decompositions, it indicates the type of interval in which
                           the eigenvalues will be found.
                             (Default value is: A)

--srange <value>           A = all singular values, V = in (vl, vu], I = from the il-th to the iu-th.
                           For partial singular value decompositions, it indicates the type of interval in which
                           the singular values will be found.
                             (Default value is: A)

--vl <value>               Lower bound of half-open interval (vl, vu].
                           Used in partial eigenvalue decomposition functions.
                           Note: the used random input matrices have all eigenvalues in [-20, 20].
                            

--vu <value>               Upper bound of half-open interval (vl, vu].
                           Used in partial eigenvalue decomposition functions.
                           Note: the used random input matrices have all eigenvalues in [-20, 20].
                            

--max_sweeps <value>       Maximum number of sweeps/iterations.
                           Used in iterative Jacobi functions.
                             (Default value is: 100)

--esort <value>            N = no sorting, A = ascending order.
                           Indicates whether the computed eigenvalues are sorted in ascending order.
                           Used in iterative Jacobi functions.
                             (Default value is: A)

--abstol <value>           Absolute tolerance at which convergence is accepted.
                           Used in iterative Jacobi and partial eigenvalue decomposition functions.
                             (Default value is: 0)

--direct <value>           F = forward, B = backward.
                           The order in which a series of transformations are applied.
                             (Default value is: F)

--pivot <value>            V = variable, T = top, B = bottom.
                           Defines the planes on which a sequence of rotations is applied.
                             (Default value is: V)

--evect <value>            N = none, V = compute eigenvectors of the matrix,
                           I = compute eigenvectors of the tridiagonal matrix.
                           Indicates how the eigenvectors are to be calculated and stored.
                             (Default value is: N)

--fast_alg <value>         O = out-of-place, I = in-place.
                           Enables out-of-place computations.
                             (Default value is: O)

--itype <value>            1 = Ax, 2 = ABx, 3 = BAx.
                           Problem type for generalized eigenproblems.
                             (Default value is: 1)

--side <value>             L = left, R = right.
                           The side from which a matrix should be multiplied.
                            

--storev <value>           C = column-wise, R = row-wise.
                           Indicates whether data is stored column-wise or row-wise.
                            

--trans <value>            N = no transpose, T = transpose, C = conjugate transpose.
                           Indicates if a matrix should be transposed.
                             (Default value is: N)

--transA <value>           N = no transpose, T = transpose, C = conjugate transpose.
                           Indicates if matrix A should be transposed.
                             (Default value is: N)

--transB <value>           N = no transpose, T = transpose, C = conjugate transpose.
                           Indicates if matrix B should be transposed.
                             (Default value is: N)

--uplo <value>             U = upper, L = lower.
                           Indicates where the data for a triangular or symmetric/hermitian matrix is stored.
                             (Default value is: U)

function implementation context:

FUNCTION_IMPLEMENTATION_CONTEXT

'''