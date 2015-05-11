package id.ac.itb.lumen.nlu.sentiment;

import java.io.Serializable;

/**
 * Bayesian network probabilistic state.
 * Created by ceefour on 22/04/2015.
 */
public class ProbabilisticState {
    public static final String TRUE = "T";
    public static final String FALSE = "F";

    private ProbabilisticVariable variable;
    private String name;

    public ProbabilisticState() {
    }

    public ProbabilisticState(ProbabilisticVariable variable, String name) {
        this.variable = variable;
        this.name = name;
    }

    /**
     * Parent probabilistic variable where this state is contained.
     * @return
     */
    public ProbabilisticVariable getVariable() {
        return variable;
    }

    public void setVariable(ProbabilisticVariable variable) {
        this.variable = variable;
    }

    /**
     * Name of probabilistic state, usually either {@link #TRUE} or {@link #FALSE}.
     * @return
     */
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ProbabilisticState that = (ProbabilisticState) o;

        if (variable != null ? !variable.equals(that.variable) : that.variable != null) return false;
        return !(name != null ? !name.equals(that.name) : that.name != null);

    }

    @Override
    public int hashCode() {
        int result = variable != null ? variable.getName().hashCode() : 0;
        result = 31 * result + (name != null ? name.hashCode() : 0);
        return result;
    }

    @Override
    public String toString() {
        return variable.getName() + ":" + name;
    }
}
