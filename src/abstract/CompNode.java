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
    public CompNode(List<Node> children, List<Node> parents) {
        super(children, parents);
    }
    abstract void forward(double[][] input);
    abstract void backward(double[][] loss);
}
