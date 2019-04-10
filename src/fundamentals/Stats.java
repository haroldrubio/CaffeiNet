public class Stats {

    /**
     * Given a set of values, perform a simple normalization on the values
     * where each element is normalized by the sum of all the elements
     * @param values A double array
     * @return A normalized double array
     */
    public static double[] normalize(double[] values){
        int numValues = values.length;
        double total = 0;
        double[] result = new double[numValues];
        for(int i = 0; i < numValues; i++) total += values[i];
        for(int i = 0; i < numValues; i++) result[i] = (values[i] / total);
        return result;
    }

    /**
     * Given a set of values, perform a softmax normalization on the values
     * where each element is gets its exponentiated value divided by the sum of all
     * exponentiated values
     * @param values A double array
     * @return A softmax normalized double array
     */
    public static double[] softmax(double[] values){
        int numValues = values.length;
        double normalizer = 0;
        double[] result = new double[numValues];
        for(int i = 0; i < numValues; i++) normalizer += (Math.exp(values[i]));
        for(int i = 0; i < numValues; i++) result[i] = (Math.exp(values[i])) / normalizer;
        return result;
    }

    /**
     * Given a probability array and events that are indexed by the array's index,
     * return the index of a uniformly random sample
     * @param probability A double array of probabilities
     * @return
     */
    public static int sampleProbability(double[] probability){
        int numEvents = probability.length;
        int event = 0;
        double[] cdf = new double[numEvents];
        double sample = Math.random();
        cdf[0] = probability[0];
        for(int i = 1; i < numEvents; i++) cdf[i] = probability[i] + cdf[i - 1];
        while(sample > cdf[event]) event++;
        return event;
    }

}
