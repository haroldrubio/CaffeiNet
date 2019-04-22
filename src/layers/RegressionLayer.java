import java.util.ArrayList;
import java.util.List;
public class RegressionLayer extends OutputLayer{
    public RegressionLayer(List<Node> children, List<Node> parents, int hiddenDim, int outputDim, double lr, double mu, double decay){
        //-----------------------Construct partial graph-----------------------
        super(children, parents, hiddenDim, outputDim, lr, mu, decay);
        lossNode = new SquaredLossNode(null, null);
        //-----------------------Construct partial graph-----------------------

        //-----------------------Connect remaining parts of the graph-----------------------
        ArrayList<Node> lossParents = new ArrayList<>();
        lossParents.add(hiddenLayer);
        ArrayList<Node> lossChildren = new ArrayList<>();
        hiddenLayer.getChildren().add(lossNode);
        lossNode.setParents(lossParents); lossNode.setChildren(lossChildren);
        this.setChildren(lossChildren);
        //-----------------------Connect remaining parts of the graph-----------------------
    }
}
