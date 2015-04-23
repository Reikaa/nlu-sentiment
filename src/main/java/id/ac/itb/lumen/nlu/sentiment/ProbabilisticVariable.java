package id.ac.itb.lumen.nlu.sentiment;

import com.google.common.collect.ImmutableList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by ceefour on 22/04/2015.
 */
public class ProbabilisticVariable {
    private static final Logger log = LoggerFactory.getLogger(ProbabilisticVariable.class);
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

    /**
     * Get probability of a state from all dependency probability values.
     * e.g. from http://en.wikipedia.org/wiki/Bayes%27_theorem :
     * P(nasional) = P(nasional | user=dakwatuna) P(user=dakwatuna) + P(nasional | user=farhatabbaslaw) P(user=farhatabbaslaw)
     * @param state
     * @return
     */
    public double getStateProbability(ProbabilisticState state) {
        if (getDependencies().isEmpty()) {
            return getProbabilities().get(ImmutableList.of(state));
        } else if (getDependencies().size() == 1) { // TODO: it's assumed we have at most one dependency
            double prob = 0.0;
            String s1 = "";
            String s2 = "";
            for (final ProbabilisticState depState : getDependencies().get(0).getStates()) {
                prob += getProbabilities().get(ImmutableList.of(depState, state)) * depState.getVariable().getStateProbability(depState);
                if (!s1.isEmpty()) {
                    s1 += " + ";
                    s2 += " + ";
                }
                s1 += "P" + ImmutableList.of(depState, state) + " P(" + depState + ")";
                s2 += getProbabilities().get(ImmutableList.of(depState, state)) + " * " + depState.getVariable().getStateProbability(depState);
            }
            log.trace("getStateProbability {}> {} = {} = {}", state, s1, s2, prob);
            return prob;
        } else {
            throw new RuntimeException("Cannot support more than 1 dependency for " + this);
        }
    }

    @Override
    public String toString() {
        return name + " (" + states.stream().map(it -> it.getName()).collect(Collectors.joining("|")) + ") {" +
                probabilities +
                ", dependencies=" + dependencies.size() +
                '}';
    }
}