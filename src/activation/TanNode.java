import java.util.List;

/** A non-linear activation function that is applied element wise
 *  and squishes values to [-1, 1]
 */
public class TanNode extends ActivationNode{
    public TanNode(List<Node> children, List<Node> parents){
        super(children, parents);
        this.f1 = new OneVarFunction() {
            @Override
            public double[][] f(double[][] input) {
                return elementWiseTan(input);
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
     * Given a mini-batch of the previous hidden layer, applies the tanh function element wise
     * @param input A double matrix
     * @return Squished values of the input matrix
     */
    private double[][] elementWiseTan(double[][] input){
        int numRows = input.length, numCols = input[0].length;
        double[][] tanVals = new double[numRows][numCols];
        for(int i = 0; i < numRows; i++) {
            for(int j = 0; j < numCols; j++){
                tanVals[i][j] = Math.tanh(input[i][j]);
            }
        }
        return tanVals;
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
                elementDerivative = 1 - Math.pow(this.hiddenState[i][j], 2);
                nextLoss[i][j] = loss[i][j] * elementDerivative;
            }
        }
        return nextLoss;
    }

}
