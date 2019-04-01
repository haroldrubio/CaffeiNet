import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Random;

class MatrixOpsTest {
    int numTests = 100;

    @Test
    void dot() {
        int dims = 3;
        int MAX_VAL = 10;
        Random rand = new Random();
        double[] vector1 = new double[dims];
        double[] vector2 = new double[dims];
        double result;
        double pred;
        for(int i = 0; i < numTests; i++){
            for(int j = 0; j < dims; j++){
                vector1[j] = MAX_VAL * rand.nextDouble();
                vector2[j] = MAX_VAL * rand.nextDouble();
            }
            result = MatrixOps.dot(vector1, vector2);
            pred = vector1[0] * vector2[0] +
                    vector1[1] * vector2[1] +
                    vector1[2] * vector2[2];
            assertEquals(result, pred);
        }
    }

    @Test
    void rightMultiply() {
        int dims = 2;
        int MAX_VAL = 10;
        Random rand = new Random();
        double[][] matrix = new double[dims][dims];
        double[] vector = new double[dims];
        double[] result;
        double[] pred = new double[dims];
        for(int i = 0; i < numTests; i++){
            for(int j = 0; j < dims; j++) vector[j] = MAX_VAL * rand.nextDouble();
            for(int j = 0; j < dims; j++)
                for(int k = 0; k < dims; k++)
                    matrix[j][k] = MAX_VAL * rand.nextDouble();
            result = MatrixOps.rightMultiply(matrix, vector);
            pred[0] = matrix[0][0] * vector[0] +
                    matrix[0][1] * vector[1];
            pred[1] = matrix[1][0] * vector[0] +
                    matrix[1][1] * vector[1];
            assertEquals(result[0], pred[0]);
            assertEquals(result[1], pred[1]);
        }
    }

    @Test
    void leftMultiply() {
        int dims = 2;
        int MAX_VAL = 10;
        Random rand = new Random();
        double[][] matrix = new double[dims][dims];
        double[] vector = new double[dims];
        double[] result;
        double[] pred = new double[dims];
        for(int i = 0; i < numTests; i++){
            for(int j = 0; j < dims; j++) vector[j] = MAX_VAL * rand.nextDouble();
            for(int j = 0; j < dims; j++)
                for(int k = 0; k < dims; k++)
                    matrix[j][k] = MAX_VAL * rand.nextDouble();
            result = MatrixOps.leftMultiply(vector, matrix);
            pred[0] = matrix[0][0] * vector[0] +
                    matrix[1][0] * vector[1];
            pred[1] = matrix[0][1] * vector[0] +
                    matrix[1][1] * vector[1];
            assertEquals(result[0], pred[0]);
            assertEquals(result[1], pred[1]);
        }
    }

    @Test
    void matrixMatrix() {
        int dims = 2;
        int MAX_VAL = 10;
        Random rand = new Random();
        double[][] matrix1 = new double[dims][dims];
        double[][] matrix2 = new double[dims][dims];
        double[][] result;
        double[][] pred = new double[dims][dims];
        for(int i = 0; i < numTests; i++){
            for(int j = 0; j < dims; j++)
                for(int k = 0; k < dims; k++){
                    matrix1[j][k] = MAX_VAL * rand.nextDouble();
                    matrix2[j][k] = MAX_VAL * rand.nextDouble();
                }
            result = MatrixOps.matrixMatrix(matrix1, matrix2);
            pred[0][0] = matrix1[0][0] * matrix2[0][0] +
                       matrix1[0][1] * matrix2[1][0];
            pred[0][1] = matrix1[0][0] * matrix2[0][1] +
                    matrix1[0][1] * matrix2[1][1];
            pred[1][0] = matrix1[1][0] * matrix2[0][0] +
                    matrix1[1][1] * matrix2[1][0];
            pred[1][1] = matrix1[1][0] * matrix2[0][1] +
                    matrix1[1][1] * matrix2[1][1];
            assertEquals(result[0][0], pred[0][0]);
            assertEquals(result[0][1], pred[0][1]);
            assertEquals(result[1][0], pred[1][0]);
            assertEquals(result[1][1], pred[1][1]);
        }
    }

    @Test
    void transpose() {
        int dims = 2;
        int MAX_VAL = 10;
        Random rand = new Random();
        double[][] matrix = new double[dims][dims];
        double[] vector = new double[dims];
        double[][] result;
        for(int i = 0; i < numTests; i++){
            for(int j = 0; j < dims; j++)
                for(int k = 0; k < dims; k++)
                    matrix[j][k] = MAX_VAL * rand.nextDouble();
            result = MatrixOps.transpose(matrix);
            assertEquals(result[0][1], matrix[1][0]);
            assertEquals(result[1][0], matrix[0][1]);
        }
    }
}