import java.util.List;

/** Similar to an input node such that it has no parents but contains variables corresponding to parameters
 *  of the overarching neural network
 *
 *  Supports vectorized representation of the input by accounting for batch size
 *  Supports mini-batch gradient descent
 */
public class ParamNode extends Node {
    private double[][] currentParameters;
    private double[][] pendingUpdate;
    private int numberOfExamples;
    public ParamNode(List<Node> children, double[][] currentParameters) {
        super(children, null);
        int numRows = currentParameters.length, numCols = currentParameters[0].length;
        this.currentParameters = currentParameters;
        this.pendingUpdate = new double[numRows][numCols];
        this.numberOfExamples = 0;
    }
    public double[][] getCurrentParameters(){
        return currentParameters;
    }

    /**
     * Caches the gradient of the loss with respect to this set of parameters
     * @param currentStep Loss gradient
     * @param batchSize An integer
     */
    public void passUpdate(double[][] currentStep, int batchSize){
        int inputDimensions = currentParameters.length;
        int outputDimensions = currentParameters[0].length;
        for(int i = 0; i < inputDimensions; i++){
            for(int j = 0; j < outputDimensions; j++){
                pendingUpdate[i][j] += currentStep[i][j];
            }
        }
        numberOfExamples = batchSize;
    }

    /**
     * Performs the gradient descent update with the learning rate passed as an argument
     * @param learningRate A double
     */
    public void updateParameters(double learningRate){
        int inputDimensions = currentParameters.length;
        int outputDimensions = currentParameters[0].length;
        for(int i = 0; i < inputDimensions; i++){
            for(int j = 0; j < outputDimensions; j++){
                currentParameters[i][j] -= learningRate * (pendingUpdate[i][j] / numberOfExamples);
                pendingUpdate[i][j] = 0;
            }
        }
        numberOfExamples = 0;
    }
}
