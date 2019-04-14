import java.util.ArrayList;
import java.util.List;

public abstract class OutputLayer extends CompNode{
    /**
     * An arbitary layer that performs a projection into the output space with an underspecified loss function
     */
    protected int hiddenDim, outputDim;
    protected double lr, mu, decay;
    protected double[][] hidden;
    protected ParamNode hiddenParameters;
    protected WeightNode hiddenLayer;
    protected LossNode lossNode;
    public OutputLayer(List<Node> children, List<Node> parents, int hiddenDim, int outputDim, double lr, double mu, double decay){
        super(children, parents);

        this.outputDim = outputDim;
        this.hiddenDim = hiddenDim + 1;
        this.lr = lr;
        this.mu = mu;
        this.decay = decay;
        //-----------------------Initialize network variables-----------------------
        hidden = MatrixOps.randomMatrix(hiddenDim + 1, outputDim);
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

        //Note: Loss gets defined in concrete subclass
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
     * Passes the loss through the layer by passing it to the loss's backward method
     * @param correct A double matrix
     */
    public void backward(double[][] correct){
        lossNode.backward(correct);
    }

    /**
     * Prints the parameter's corresponding to the hidden layer's parameter node
     */
    public void printParameters(){
        hiddenLayer.printParameters();
    }

    public void printLoss(){
        lossNode.printLoss();
    }

}

