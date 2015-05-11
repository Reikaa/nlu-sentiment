package id.ac.itb.lumen.nlu.sentiment;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Represents a Bayesian network probabilistic graph model.
 * Created by ceefour on 22/04/2015.
 */
public class BayesianNetwork {

    private List<ProbabilisticVariable> variables = new ArrayList<>();

    /**
     * Probabilistic variables that make up the entire graph.
     * @return
     */
    public List<ProbabilisticVariable> getVariables() {
        return variables;
    }

    /**
     * Get an existing {@link ProbabilisticVariable} by name,
     * or create new variable if not already exists.
     * @param name
     * @return
     */
    public ProbabilisticVariable getOrCreateVariable(String name) {
        final Optional<ProbabilisticVariable> existing = variables.stream().filter(it -> name.equals(it.getName())).findAny();
        if (existing.isPresent()) {
            return existing.get();
        } else {
            final ProbabilisticVariable variable = new ProbabilisticVariable();
            variables.add(variable);
            variable.setName(name);
            return variable;
        }
    }

    @Override
    public String toString() {
        return "BayesianNetwork{" +
                "variables(" + variables.size() + ")=" + variables.stream().limit(10).toArray() +
                '}';
    }

    /**
     * Prints the entire Bayesian network and its {@link ProbabilisticVariable}s,
     * useful for debugging.
     * @return
     */
    public String toStringComplete() {
        String s = "BayesianNetwork (" + variables.size() + " variables) {\n";
        for (final ProbabilisticVariable v : variables) {
            s += "  " + v + "\n";
        }
        return s;
    }

}
