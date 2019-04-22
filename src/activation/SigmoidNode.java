import java.util.List;
/** A non-linear activation function that is applied element wise
 *  and squishes values to [0, 1]
 */
public class SigmoidNode extends ActivationNode{
    public SigmoidNode(List<Node> children, List<Node> parents){
        super(children, parents);
        this.f1 = new OneVarFunction() {
            @Override
            public double[][] f(double[][] input) {
                return elementWiseSig(input);
            }
        };
        this.derivatives1 = new OneVarDerivative[]{
                new OneVarDerivative() {
                    @Override
                    public double[][] deriv(double[][] loss) {
                        return elementWiseDeriv(loss);
                    }
                }
        };
    }

    /**
     * Given a mini-batch of the previous hidden layer, applies the sigmoid function element wise
     * @param input A double matrix
     * @return Squished values of the input matrix
     */
    private double[][] elementWiseSig(double[][] input){
        int numRows = input.length, numCols = input[0].length;
        double[][] sigVals = new double[numRows][numCols];
        for(int i = 0; i < numRows; i++) {
            for(int j = 0; j < numCols; j++){
                sigVals[i][j] = 1/(1 + Math.exp(-1*input[i][j]));
            }
        }
        return sigVals;
    }
    /**
     * Given the loss from a child node, multiply each element by its corresponding derivative
     * @param loss A double matrix
     * @return Derivative of loss with respect to the TanNode
     */
    private double[][] elementWiseDeriv(double[][] loss){
        int numRows = this.hiddenState.length, numCols = this.hiddenState[0].length;
        double[][] nextLoss = new double[numRows][numCols];
        double elementDerivative;
        for(int i = 0; i < numRows; i++){
            for(int j = 0; j < numCols; j++){
                elementDerivative = this.hiddenState[i][j] * (1 - this.hiddenState[i][j]);
                nextLoss[i][j] = loss[i][j] * elementDerivative;
            }
        }
        return nextLoss;
    }
}
