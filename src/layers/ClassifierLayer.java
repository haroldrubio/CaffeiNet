import java.util.ArrayList;
import java.util.List;

/**
 * A layer that performs classification
 * Contains a projection into the output space, a softmax normalization and the computation
 * of cross entropy loss
 */
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

    /**
     * Given one-hot encoded vectors for classification, gets the accuracy of the predictions
     * made by the network
     *
     * These predictions are obtained from taking an arg max over the softmax probabilities
     * @param correct A double matrix of correct labels
     * @return A double of the accuracy of the predictions
     */
    public double getAccuracy(double[][] correct){
        int numExamples = correct.length, numClasses = correct[0].length;
        int numCorrect = 0;
        double[][] predictions = lossNode.predictions;
        double[] predictedClasses = new double[numExamples];
        double currentMax;
        for(int i = 0; i < numExamples; i++){
            currentMax = 0;
            for(int k = 0; k < numClasses; k++)
                if(currentMax <= predictions[i][k]){predictedClasses[i] = k; currentMax = predictions[i][k];}
        }
        for(int i = 0; i < numExamples; i++)
            for(int k = 0; k < numClasses; k++)
                if(correct[i][k] > 0 && predictedClasses[i] == k){numCorrect++; break;}
        return (double)numCorrect/(double)numExamples;

    }
}
