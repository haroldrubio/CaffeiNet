import java.util.List;

/** A non-linear activation function that is applied element wise
 *  and squishes values to [-1, 1]
 */
public class TanNode extends CompNode{
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

    /**
     * Computes this node's function on the input, caches the values and passes
     * its value to its children
     * @param input A double matrix
     */
    public void forward(double[][] input){
        this.hiddenState = this.f1.f(input);
        this.prevHidden = input;
        CompNode nextComp;
        List<Node> currentChildren = this.getChildren();
        if(currentChildren != null)
            for(Node n: getChildren()){
                if(n instanceof CompNode){
                    nextComp = (CompNode) n;
                    nextComp.forward(hiddenState);
                }
            }
    }

    /**
     * Performs backpropagation by applying this node's derivative function on the loss
     *
     * Keep passing a null gradient if null was received in order to signal a parameter update
     * @param loss A double matrix
     */
    public void backward(double[][] loss){
        CompNode nextComp;
        List<Node> currentParents = this.getParents();
        double[][] nextGradient = null;
        //Send the signal backwards
        if(loss != null)
            nextGradient = this.derivatives1[0].deriv(loss);
        if(currentParents != null)
            for(Node n: currentParents)
                if(n instanceof CompNode) {
                    nextComp = (CompNode) n;
                    nextComp.backward(nextGradient);
                }
    }


}
