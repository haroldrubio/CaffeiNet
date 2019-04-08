import java.util.List;
public class InputNode extends Node{
    public InputNode(List<Node> children){
        super(children, null);
    }
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
