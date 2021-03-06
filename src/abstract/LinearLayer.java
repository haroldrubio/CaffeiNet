import java.util.ArrayList;
import java.util.List;

public abstract class LinearLayer extends CompNode{
    /**
     * A feed-forward layer that performs the calculation Y = f(XW) given a mini-batch of examples X
     * Encapsulates a WeightNode, ParamNode and TanNode and facilitates the forward and backward
     * passes through its components
     */
    protected int inputDim, hiddenDim;
    protected double lr, mu, decay;
    protected double[][] hidden;
    protected ParamNode hiddenParameters;
    protected WeightNode hiddenLayer;
    protected ActivationNode nonlinearity;
    public LinearLayer(List<Node> children, List<Node> parents, int inputDim, int hiddenDim, double lr, double mu, double decay){
        super(children, parents);

        this.inputDim = inputDim + 1;
        this.hiddenDim = hiddenDim;
        this.lr = lr;
        this.mu = mu;
        this.decay = decay;
        //-----------------------Initialize network variables-----------------------
        hidden = MatrixOps.randomMatrix(inputDim + 1, hiddenDim);
        hiddenParameters = new ParamNode(null, hidden);
        hiddenLayer = new WeightNode(null, null, lr, mu, decay);
        //-----------------------Initialize network variables-----------------------

        //-----------------------Connect Computational Graph-----------------------
        ArrayList<Node> hiddenParents = new ArrayList<>();
        hiddenParents.add(hiddenParameters);
        ArrayList<Node> hiddenChildren = new ArrayList<>();
        hiddenLayer.setParents(hiddenParents); hiddenLayer.setChildren(hiddenChildren);
        //-----------------------Connect Computational Graph-----------------------

        //-----------------------Connect Overarching Input-----------------------
        this.setParents(hiddenParents);
        //-----------------------Connect Overarching Input-----------------------

        //Note: Activation gets defined in concrete subclass
    }

    /**
     * Expand the current input to allow for a bias term
     * Passes augmented input to the hidden layer node
     * @param input A double matrix
     */
    public void forward(double[][] input){
        //Add bias to input
        int numRows = input.length, numCols = input[0].length + 1;
        int biasIndex = numCols - 1;
        double[][] augmentedInput = new double[numRows][numCols];
        for(int i = 0; i < numRows; i++){
            for(int j = 0; j < numCols - 1; j++){
                augmentedInput[i][j] = input[i][j];
            }
            augmentedInput[i][biasIndex] = 1;
        }

        //Pass to hidden layer
        hiddenLayer.forward(augmentedInput);
    }

    /**
     * Passes the loss through the layer by passing it to the nonlinearity's backward method
     * @param loss A double matrix
     */
    public void backward(double[][] loss){
        nonlinearity.backward(loss);
    }

    /**
     * Sets this node's learning rate
     * @param learningRate A double
     */
    public void setLearningRate(double learningRate) {
        this.hiddenLayer.setLearningRate(learningRate);
    }

    /**
     * Sets this node's momentum coefficient
     * @param momentum A double
     */
    public void setMomentum(double momentum) {
        this.hiddenLayer.setMomentum(momentum);
    }

    /**
     * Set this node's weight decay coefficient
     * @param decay A double
     */
    public void setDecay(double decay) {
        this.hiddenLayer.setDecay(decay);
    }

    /**
     * Prints the parameter's corresponding to the hidden layer's parameter node
     */
    public void printParameters(){
        hiddenLayer.printParameters();
    }

}

