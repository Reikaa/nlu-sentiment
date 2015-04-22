package id.ac.itb.lumen.nlu.sentiment;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Created by ceefour on 22/04/2015.
 */
public class BayesianNetwork {

    private List<ProbabilisticVariable> variables = new ArrayList<>();

    public List<ProbabilisticVariable> getVariables() {
        return variables;
    }

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

    public String toStringComplete() {
        String s = "BayesianNetwork (" + variables.size() + " variables) {\n";
        for (final ProbabilisticVariable v : variables) {
            s += "  " + v + "\n";
        }
        return s;
    }

}
