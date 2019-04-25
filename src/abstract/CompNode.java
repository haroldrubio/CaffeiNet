import java.util.List;

/** A node in the computational graph that performs an operation
 *
 *  Supports computations of functions of up to two variables
 *
 *  Provides abstract methods that allow for the forward and backward propagation
 *  through the graph through this node
 */
public abstract class CompNode extends Node{
    protected OneVarFunction f1;
    protected TwoVarFunction f2;
    protected OneVarDerivative[] derivatives1;
    protected TwoVarDerivative[] derivatives2;
    protected double[][] hiddenState;
    protected double[][] prevHidden;
    public CompNode(List<Node> children, List<Node> parents) {
        super(children, parents);
    }
    abstract void forward(double[][] input);
    abstract void backward(double[][] loss);

    /**
     * Passes a double matrix, usually a loss matrix, to the parents as the backwards step
     * @param loss A double matrix
     */
    protected void passToParents(double[][] loss){
        CompNode nextComp;
        List<Node> currentParents = this.getParents();
        if(currentParents != null)
            for(Node n: currentParents)
                if(n instanceof CompNode) {
                    nextComp = (CompNode) n;
                    nextComp.backward(loss);
                }
    }

    /**
     * Passes a double matrix, usually the hidden state of the current node, to the children as the forwards step
     * @param nextValue A double matrix
     */

    protected void passToChildren(double[][] nextValue){
        CompNode nextComp;
        List<Node> currentChildren = this.getChildren();
        if(currentChildren != null)
            for(Node n: getChildren()){
                if(n instanceof CompNode){
                    nextComp = (CompNode) n;
                    nextComp.forward(nextValue);
                }
            }
    }

    public double[][] getHiddenState(){
        return this.hiddenState;
    }

    public double[][] getPrevHidden(){
        return this.prevHidden;
    }
}
