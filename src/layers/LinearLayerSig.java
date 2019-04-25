import java.util.ArrayList;
import java.util.List;

/**
 * A linear layer using the sigmoid non-linearity
 */
public class LinearLayerSig extends LinearLayer{
    public LinearLayerSig(List<Node> children, List<Node> parents, int inputDim, int hiddenDim, double lr, double mu, double decay){
        //-----------------------Construct partial graph-----------------------
        super(children, parents, inputDim, hiddenDim, lr, mu, decay);
        nonlinearity = new SigmoidNode(null, null);
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
