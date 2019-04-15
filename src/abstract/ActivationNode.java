import java.util.List;

public abstract class ActivationNode extends CompNode{
    public ActivationNode(List<Node> children, List<Node> parents){
        super(children, parents);
    }

    /**
     * Computes this node's function on the input, caches the values and passes
     * its value to its children
     * @param input A double matrix
     */
    public void forward(double[][] input){
        this.hiddenState = this.f1.f(input);
        this.prevHidden = input;
        super.passToChildren(hiddenState);
    }

    /**
     * Performs backpropagation by applying this node's derivative function on the loss
     *
     * Keep passing a null gradient if null was received in order to signal a parameter update
     * @param loss A double matrix
     */
    public void backward(double[][] loss){
        double[][] nextGradient = null;
        //Send the signal backwards
        if(loss != null)
            nextGradient = this.derivatives1[0].deriv(loss);
        super.passToParents(nextGradient);
    }
}
