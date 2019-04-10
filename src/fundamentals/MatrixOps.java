public class MatrixOps {
    /**
     * Computes the scalar product between the first and second vectors
     * Assumes the dimensionality of the vectors are the same
     * @param vector1 The first argument to the dot product
     * @param vector2 The second argument to the dot product
     * @return vector1 dotted with vector2
     */
    public static double dot(double[] vector1, double[] vector2){
        double result = 0;
        int dims = vector1.length;
        for(int i = 0; i < dims; i++)
            result += (vector1[i] * vector2[i]);
        return result;
    }

    /**
     * Given matrix A and column vector x, computes the product Ax, which is the mapping of x from
     * the column space of A into the row space of A
     * Assumes the number of columns in the matrix is equal to the number of rows in the matrix
     * @param matrix A double matrix
     * @param vector A double vector
     * @return A column vector mapped into the row space of the matrix
     */
    public static double[] rightMultiply(double[][] matrix, double[] vector){
        int outputDims = matrix.length;
        double[] result = new double[outputDims];
        for(int i = 0; i < outputDims; i++)
            result[i] = MatrixOps.dot(matrix[i], vector);
        return result;
    }

    /**
     * Given a row vector x and matrix A, computes the product xA which is the mapping of x from
     * the row space of A into the column space of A
     * Assume the number of columns in the vector is equal to the number of the rows in the matrix
     * @param vector A double vector
     * @param matrix A double matrix
     * @return A row vector mapped into the column space of the matrix
     */
    public static double[] leftMultiply(double[] vector, double[][] matrix){
        int outputDims = matrix[0].length;
        double[] result = new double[outputDims];
        double[] currentColumn;
        for(int i = 0; i < outputDims; i++){
            currentColumn = MatrixOps.getColumn(matrix, i);
            result[i] = MatrixOps.dot(vector, currentColumn);
        }
        return result;
    }

    /**
     * Given matricies A and B, performs the operation AB by computing a series of dot products
     * @param matrix1 A double matrix
     * @param matrix2 A double matrix
     * @return A double matrix corresponding to the matrix product
     */
    public static double[][] matrixMatrix(double[][] matrix1, double[][] matrix2){
        int newRowDim = matrix1.length;
        int newColDim = matrix2[0].length;
        matrix2 = MatrixOps.transpose(matrix2);
        double[][] result = new double[newRowDim][newColDim];
        for(int i = 0; i < newRowDim; i++){
            for(int j = 0; j < newColDim; j++){
                result[i][j] = MatrixOps.dot(matrix1[i], matrix2[j]);
            }
        }
        return result;
    }

    /**
     * Given a vector A, normalizes the vector returns a unit vector pointing in the direction of A
     * @param vector A double vector
     * @return A double unit vector
     */
    public static double[] normalize(double[] vector){
        int vectorDims = vector.length;
        double norm = MatrixOps.l2Norm(vector);
        double[] normalizedVec = new double[vectorDims];
        for(int i = 0; i < vectorDims; i++){
            normalizedVec[i] = vector[i] / norm;
        }
        return normalizedVec;
    }

    /**
     * Given vector A, get the length of the vector according to the l2 norm
     * @param vector A double vector
     * @return The l2 norm
     */
    public static double l2Norm(double[] vector){
        int vectorDims = vector.length;
        double norm = 0;
        for(int i = 0; i < vectorDims; i++){
            norm += Math.pow(vector[i], 2);
        }
        return Math.sqrt(norm);
    }

    /**
     * Given matrix A, return A^T, which swaps the rows and columns of A
     * @param matrix A double matrix
     * @return The transpose of the given matrix
     */
    public static double[][] transpose(double[][] matrix){
        int rowDim = matrix.length;
        int columnDim = matrix[0].length;
        double[][] result = new double[columnDim][rowDim];
        for(int i = 0; i < columnDim; i++)
            result[i] = getColumn(matrix, i);
        return result;
    }

    /**
     * Given a matrix and a zero-indexed column number, return the column of the given matrix
     * @param matrix A double matrix
     * @param column An integer index
     * @return A double array containing the column at the 'column' index
     */
    private static double[] getColumn(double[][] matrix, int column){
        int rowDim = matrix.length;
        double[] result = new double[rowDim];
        for(int i = 0; i < rowDim; i++){
            result[i] = matrix[i][column];
        }
        return result;
    }

    /**
     * Constructs a matrix with a specified number of rows and columns
     * @param numRows An int
     * @param numCols An int
     * @return A matrix with entries in the interval (-1.0, 1.0)
     */
    public static double[][] randomMatrix(int numRows, int numCols){
        double[][] randMatrix = new double[numRows][numCols];
        double flipSign, randValue;
        for(int i = 0; i < numRows; i++){
            for(int j = 0; j < numCols; j++){
                flipSign = Math.random(); randValue = Math.random();
                randMatrix[i][j] = (flipSign > 0.5) ? randValue : (-1*randValue);
            }
        }
        return randMatrix;
    }

    /**
     * Constructs a vector with a specified dimensionality
     * @param dimensions An int
     * @return A vector with entries in the interval (-1.0, 1.0)
     */
    public static double[] randomVector(int dimensions){
        double[] randVector = new double[dimensions];
        double flipSign, randValue;
        for(int i = 0; i < dimensions; i++){
                flipSign = Math.random(); randValue = Math.random();
                randVector[i] = (flipSign > 0.5) ? randValue : (-1*randValue);
        }
        return randVector;
    }

    /**
     * Given vectors A and B in this order, computes the difference A - B component wise
     * @param vector1 A double vector
     * @param vector2 A double vector
     * @return Difference between the two arguments
     */
    public static double[] difference(double[] vector1, double[] vector2){
        int numDims = vector1.length;
        double[] diffVector = new double[numDims];
        for(int i = 0; i < numDims; i++)
            diffVector[i] = vector1[i] - vector2[i];
        return diffVector;
    }

}
