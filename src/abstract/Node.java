import java.util.List;

/** Parent class to all other classes used in the computational graph
 *  Node follows the standard representation of a node in any graph, containing
 *  adjacency list to its parents and children such that the union of the two lists
 *  forms the total adjacency list of its neighbors
 *
 *  Provides definitions for interfaces that support forward and backward
 *  propagation through the network
 *
 *  Allows the grouping of multiple computational graph nodes into a single node
 */
public abstract class Node {
    private List<Node> children;
    private List<Node> parents;
    protected interface OneVarFunction{
        double[][] f(double[][] input);
    }
    protected interface TwoVarFunction{
        double[][] f(double[][] input1, double[][] input2);
    }
    protected interface OneVarDerivative{
        double[][] deriv(double[][] input);
    }
    protected interface TwoVarDerivative{
        double[][] deriv(double[][] input1, double[][] input2);
    }
    public Node(List<Node> children, List<Node> parents) {
        this.children = children;
        this.parents = parents;
    }

    public List<Node> getChildren(){
        return this.children;
    }

    public List<Node> getParents(){
        return this.parents;
    }

    public void setChildren(List<Node> children){
        this.children = children;
    }

    public void setParents(List<Node> parents){
        this.parents = parents;
    }
}
