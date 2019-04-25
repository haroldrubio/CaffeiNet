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
    private double[][] currentVelocity;
    private double[][] previousVelocity;
    private int numberOfExamples;
    public ParamNode(List<Node> children, double[][] currentParameters) {
        super(children, null);
        int numRows = currentParameters.length, numCols = currentParameters[0].length;
        this.currentParameters = currentParameters;
        this.pendingUpdate = new double[numRows][numCols];
        this.currentVelocity = new double[numRows][numCols];
        this.previousVelocity = new double[numRows][numCols];
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
     * Performs the gradient descent update with the learning rate and hyperparameter passed as an argument
     * Type of gradient descent: mini-batch with Nesterov Accerlated Gradient
     * This specific form of NAG is provided by the CS231n github.io page
     * @param learningRate A double hyperparameter
     * @param momentum A double hyperparameter
     * @param decay A double hyperparameter
     */
    public void updateParameters(double learningRate, double momentum, double decay){
        int inputDimensions = currentParameters.length;
        int outputDimensions = currentParameters[0].length;
        double nextUpdate;
        for(int i = 0; i < inputDimensions; i++){
            for(int j = 0; j < outputDimensions; j++){
                previousVelocity[i][j] = currentVelocity[i][j];
                currentVelocity[i][j] = momentum * currentVelocity[i][j] - learningRate * pendingUpdate[i][j];
                nextUpdate = (1 + momentum) * currentVelocity[i][j] - momentum * previousVelocity[i][j];
                currentParameters[i][j] += (nextUpdate / numberOfExamples) - decay * currentParameters[i][j];
                //Below: Vanilla Mini-Batch GD update
                //currentParameters[i][j] -= learningRate*(pendingUpdate[i][j]/ numberOfExamples);
                pendingUpdate[i][j] = 0;
            }
        }
        numberOfExamples = 0;
    }
}
