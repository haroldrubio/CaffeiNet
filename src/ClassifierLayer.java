import java.util.ArrayList;
import java.util.List;

public class ClassifierLayer extends OutputLayer{
    protected SoftmaxNode softmax;
    public ClassifierLayer(List<Node> children, List<Node> parents, int hiddenDim, int outputDim, double lr, double mu, double decay){
        //-----------------------Construct partial graph-----------------------
        super(children, parents, hiddenDim, outputDim, lr, mu, decay);
        softmax = new SoftmaxNode(null, null);
        lossNode = new EntropyLossNode(null, null);
        //-----------------------Construct partial graph-----------------------

        //-----------------------Connect remaining parts of the graph-----------------------
        ArrayList<Node> softmaxParents = new ArrayList<>();
        softmaxParents.add(hiddenLayer);
        ArrayList<Node> softmaxChildren = new ArrayList<>();
        softmaxChildren.add(lossNode);
        ArrayList<Node> lossParents = new ArrayList<>();
        lossParents.add(softmax);
        ArrayList<Node> lossChildren = new ArrayList<>();
        hiddenLayer.getChildren().add(softmax);
        softmax.setParents(softmaxParents); softmax.setChildren(softmaxChildren);
        lossNode.setParents(lossParents); lossNode.setChildren(lossChildren);
        this.setChildren(lossChildren);
        //-----------------------Connect remaining parts of the graph-----------------------
    }
}
