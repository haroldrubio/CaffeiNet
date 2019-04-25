import java.util.List;

/**
 *  A node that indicates the termination of the computational graph
 *  Contains the final predictions as the raw value in regression, or the softmax probabilities in the case
 *  of classification
 */
public abstract class LossNode extends CompNode{
    protected double[][] predictions;
    protected double[][] batchLoss;
    public LossNode(List<Node> children, List<Node> parents){
        super(children, parents);
    }

    /**
     * Stores the value passed as the networks predictions
     * No calls to the children are performed since the LossNode terminates the graph
     * @param predicted A double matrix
     */
    public void forward(double[][] predicted){
        predictions = predicted;
    }

    /**
     * Takes the correct data as a parameter, computes the loss and backpropagates the gradient
     * into the rest of the computational graph
     * @param correct A double matrix of the correct labels
     */
    public void backward(double[][] correct){
        double[][] nextGradient = null;
        //Send the signal backwards
        if(correct != null) {
            batchLoss = this.f1.f(correct);
            nextGradient = this.derivatives1[0].deriv(correct);
        }
        super.passToParents(nextGradient);
    }

    /**
     * Standalone function to calculate the loss and assigns it to the instance variable
     * Does not perform backprop on the correct data
     * @param correct A double matrix of the correct data
     */
    protected void calculateLoss(double[][] correct){
        batchLoss = this.f1.f(correct);
    }

    /**
     * Prints the value of loss to standard out
     */
    public void printLoss(){
        if(batchLoss == null) return;
        int batchSize = batchLoss.length;
        double sumLoss = 0;
        for(int i = 0; i < batchSize; i++){
            sumLoss += Math.pow(batchLoss[i][0], 2);
        }
        sumLoss = Math.sqrt(sumLoss);
        System.out.printf("%5.3f\n", sumLoss);
    }

    /**
     * Returns the batch loss as a number where the batch loss is the L2 norm
     * of the loss vector
     * @return A double representing the loss
     */
    public double getLoss(){
        if(batchLoss == null) return -1;
        int batchSize = batchLoss.length;
        double sumLoss = 0;
        for(int i = 0; i < batchSize; i++){
            sumLoss += Math.pow(batchLoss[i][0], 2);
        }
        sumLoss = Math.sqrt(sumLoss);
        return sumLoss;
    }

}
