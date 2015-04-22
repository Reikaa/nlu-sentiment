package id.ac.itb.lumen.nlu.sentiment;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by ceefour on 22/04/2015.
 */
public class ProbabilisticVariable {
    private String name;
    private List<ProbabilisticState> states = new ArrayList<>();
    private List<ProbabilisticVariable> dependencies = new ArrayList<>();
    private Map<List<ProbabilisticState>, Double> probabilities = new LinkedHashMap<>(); // so it prints nicely

    public ProbabilisticVariable() {
    }

    public ProbabilisticVariable(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<ProbabilisticState> getStates() {
        return states;
    }

    /**
     * Other {@link ProbabilisticVariable}s that influences this variable.
     * @return
     */
    public List<ProbabilisticVariable> getDependencies() {
        return dependencies;
    }

    public Map<List<ProbabilisticState>, Double> getProbabilities() {
        return probabilities;
    }

    @Override
    public String toString() {
        return name + " (" + states.stream().map(it -> it.getName()).collect(Collectors.joining("|")) + ") {" +
                probabilities +
                ", dependencies=" + dependencies.size() +
                '}';
    }
}