import java.util.List;

/** Specific type of node that does not have any parents but supports a forward pass
 *  into the computational graph
 *
 *  Does not support a backward pass since the input does not need to be changed
 */
public class InputNode extends Node{
    public InputNode(List<Node> children){
        super(children, null);
    }

    /**
     * Given a mini-batch of examples, propagate it forward through the input node's children
     * than can perform a computation
     * @param input A double matrix
     */
    public void forward(double[][] input){
        List<Node> currentChildren = super.getChildren();
        CompNode nextComputation;
        for(Node nextNode: currentChildren){
            if(nextNode instanceof CompNode) {
                nextComputation = (CompNode) nextNode;
                nextComputation.forward(input);
            }
        }
    }
}
