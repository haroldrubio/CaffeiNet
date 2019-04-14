import java.util.List;

/** Takes a mini-batch of inputs and projects the vectors into a new space
 *  with the mapping contained in a parameter node
 *
 *  Since this node contains a parameter node, it supports features that control the type of update
 *  (see ParamNode for update details)
 */
public class WeightNode extends CompNode{
    private double[][] weights;
    private double learningRate;
    private double momentum;
    private double decay;
    private ParamNode parameters;
    public WeightNode(List<Node> children, List<Node> parents, double learningRate, double momentum, double decay){
        super(children, parents);
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.decay = decay;
        this.f1 = new OneVarFunction() {
            @Override
            public double[][] f(double[][] input) {
                return MatrixOps.matrixMatrix(input, weights);
            }
        };

        this.derivatives1 = new OneVarDerivative[] {
                //[0] is derivative with respect to X
                new OneVarDerivative() {
                    @Override
                    public double[][] deriv(double[][] input) {
                        return MatrixOps.matrixMatrix(input, MatrixOps.transpose(weights));
                    }
                },
                //[1] is derivative with respect to W
                new OneVarDerivative() {
                    @Override
                    public double[][] deriv(double[][] input) {
                        return MatrixOps.matrixMatrix(MatrixOps.transpose(prevHidden), input);
                    }
                }
        };

    }

    /**
     * Fetches the newest set of parameters from its parent parameter node
     * Projects the mini-batch into the new space with the weights and caches its arguments and calculations
     * Passes the calculated value to its children that can perform a computation
     * @param input A double matrix
     */
    public void forward(double[][] input){
        List<Node> parents = this.getParents();
        for(Node n: parents)
            if(n instanceof ParamNode){
                parameters = (ParamNode) n;
                weights = parameters.getCurrentParameters();
            }
        this.prevHidden = input;
        this.hiddenState = this.f1.f(input);
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
     * Calculates a derivative with respect to the weights and passes the update to the parameter node
     * Calculates a derivative with respect to the previous input and passes the gradient to its children
     * than can process a gradient
     *
     * Perform the passed parameter update when receiving a null reference
     * @param loss A double matrix
     */
    public void backward(double[][] loss){
        CompNode nextComp;
        List<Node> currentParents = this.getParents();
        double[][] nextGradient = null;
        if(loss == null)
            parameters.updateParameters(learningRate, momentum, decay);
        else{
            int batchSize = loss.length;
            //First send the next update to the parameters
            double[][] nextUpdate = this.derivatives1[1].deriv(loss);
            parameters.passUpdate(nextUpdate, batchSize);
            //Then send the signal backwards
            nextGradient = this.derivatives1[0].deriv(loss);
        }
        if(currentParents != null)
            for(Node n: currentParents)
                if(n instanceof CompNode) {
                    nextComp = (CompNode) n;
                    nextComp.backward(nextGradient);
                }
    }

    // These methods allow the implementation of annealing any of the gradient descent update
    // hyperparameters

    /**
     * Sets this node's learning rate
     * @param learningRate A double
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Sets this node's momentum coefficient
     * @param momentum A double
     */
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    /**
     * Set this node's weight decay coefficient
     * @param decay A double
     */
    public void setDecay(double decay) {
        this.decay = decay;
    }

    /**
     * Gets this node's hidden state
     * @return A double matrix
     */
    public double[][] getHiddenState(){
        return this.hiddenState;
    }

    /**
     * Gets the argument to this node's function (mini-batch of values)
     * @return A double matrix
     */
    public double[][] getPrevHidden(){
        return this.prevHidden;
    }

    /**
     * Prints the current weights for this node's parameters
     */
    public void printParameters(){
        int numRows = weights.length, numCols = weights[0].length;
        System.out.println("--------------Weights---------------");
        System.out.print("{");
        for(int i = 0; i < numRows; i++){
            System.out.printf("{");
            for(int j = 0; j < numCols; j++){
                if(j == (numCols - 1))
                    System.out.printf("%3.2f},\n", weights[i][j]);
                else
                    System.out.printf("%3.2f, ", weights[i][j]);
            }
        }
        System.out.print("}\n");
        System.out.println("--------------Weights---------------");
    }

}
