import java.util.ArrayList;
import java.util.List;

/**
 * A linear layer using the tanh nonlinearity
 */
public class LinearLayerTan extends LinearLayer{
    public LinearLayerTan(List<Node> children, List<Node> parents, int inputDim, int hiddenDim, double lr, double mu, double decay){
        //-----------------------Construct partial graph-----------------------
        super(children, parents, inputDim, hiddenDim, lr, mu, decay);
        nonlinearity = new TanNode(null, null);
        //-----------------------Construct partial graph-----------------------

        //-----------------------Connect remaining parts of the graph-----------------------
        ArrayList<Node> nonlinearParents = new ArrayList<>();
        nonlinearParents.add(hiddenLayer);
        ArrayList<Node> nonlinearChildren = new ArrayList<>();
        hiddenLayer.getChildren().add(nonlinearity);
        nonlinearity.setParents(nonlinearParents); nonlinearity.setChildren(nonlinearChildren);
        this.setChildren(nonlinearChildren);
        //-----------------------Connect remaining parts of the graph-----------------------
    }
}
