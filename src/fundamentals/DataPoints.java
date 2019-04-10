import java.util.Random;

/**
 * Groups data as an input matrix with dimensions of [numExamples] x [numDimensions]
 * and an output vector with dimensionality of [numExamples]
 * @author Harold Rubio
 */
public class DataPoints{
    private double[][] inputData; //Input matrix whose rows are row vectors of the input
    private int[] outputLabels; //An int array where the ith element is the class of the ith input
    private int numExamples; //The number of examples inputted so far
    private int batchIndex; //Index to indicate where the next batch will pull from
    /**
     * Allocate memory for the data but does not set the values
     * @param numPoints Number of examples
     * @param inputDims Dimensionality of the input data
     */
    public DataPoints(int numPoints, int inputDims){
        inputData = new double[numPoints][inputDims];
        outputLabels = new int[numPoints];
        numExamples = 0;
        batchIndex = 0;
    }

    /**
     * Adds an example to the data set
     * This is done to allow the user to pre-process the data as he/she wishes
     * @param sample A double array representing the input
     * @param output A int representing a class
     */
    public void setData(double[] sample, int output){
        inputData[numExamples] = sample;
        outputLabels[numExamples] = output;
        numExamples++;
    }

    /**
     * Randomly organizes the data by swapping two randomly chosen rows in the data
     */
    public void shuffle(){
        Random rand = new Random();
        int originalIndex = 0, swapIndex = 0;
        double[] tempRow;
        for(int i = 0; i < numExamples; i++){
            originalIndex = rand.nextInt(numExamples);
            while(swapIndex == originalIndex) swapIndex = rand.nextInt(numExamples);
            tempRow = inputData[originalIndex];
            inputData[originalIndex] = inputData[swapIndex];
            inputData[swapIndex] = tempRow;
        }
    }

    /**
     * Gets a sample of size batchSize from the data
     * If the batch reaches the end of the data, samples from the beginning
     * @param batchSize An integer for batchSize
     * @return A double matrix with batchSize rows containing a sample of the input
     */
    public double[][] nextBatch(int batchSize){
        int inputDims = inputData[0].length;
        double[][] batch = new double[batchSize][inputDims];
        for(int i = 0; i < batchSize; i++){
            for(int j = 0; j < inputDims; j++){
                batch[batchIndex][j] = inputData[batchIndex][j];
            }
            batchIndex = (batchIndex == (numExamples - 1)) ? 0 : (batchIndex + 1);
        }
        return batch;
    }

}
