package id.ac.itb.lumen.nlu.sentiment;

import com.google.common.collect.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.annotation.Profile;

import javax.inject.Inject;
import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

@SpringBootApplication
@Profile("nlu-sentiment")
public class NluSentimentApplication implements CommandLineRunner {

    private static Logger log = LoggerFactory.getLogger(NluSentimentApplication.class);
    private ProbabilisticVariable screenNamePv;
    private Map<String, ProbabilisticVariable> wordPvs;

    public static void main(String[] args) {
        new SpringApplicationBuilder(NluSentimentApplication.class)
                .profiles("nlu-sentiment")
                .run(args);
    }

    /**
     * key: {screenName}/{word}
     */
    protected Map<String, Double> wordNormLengthByScreenName = new LinkedHashMap<>();
    protected Set<String> allWords = new LinkedHashSet<>();

    protected SentimentAnalyzer analyze(File f, int wordLimit, Set<String> moreStopWords) {
        final SentimentAnalyzer sentimentAnalyzer = new SentimentAnalyzer();
        sentimentAnalyzer.readCsv(f);
        sentimentAnalyzer.lowerCaseAll();
        sentimentAnalyzer.removeLinks();
        sentimentAnalyzer.removePunctuation();
        sentimentAnalyzer.removeNumbers();
        sentimentAnalyzer.canonicalizeWords();
        sentimentAnalyzer.removeStopWords(moreStopWords.toArray(new String[] {}));
        log.info("Preprocessed text: {}", sentimentAnalyzer.texts.entrySet().stream().limit(10)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
        sentimentAnalyzer.splitWords();
        log.info("Words: {}", sentimentAnalyzer.words.entrySet().stream().limit(10)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));

        final ImmutableMultiset<String> wordMultiset = Multisets.copyHighestCountFirst(HashMultiset.create(
                sentimentAnalyzer.words.values().stream().flatMap(it -> it.stream()).collect(Collectors.toList())) );
        final Map<String, Integer> wordCounts = new LinkedHashMap<>();
        // only the N most used words
        wordMultiset.elementSet().stream().limit(wordLimit).forEach( it -> wordCounts.put(it, wordMultiset.count(it)) );
        log.info("Word counts (orig): {}", wordCounts);

        // Normalize the twitterUser "vector" to length 1.0
        // Note that this "vector" is actually user-specific, i.e. it's not a user-independent vector
        long origSumSqrs = 0;
        for (final Integer it : wordCounts.values()) {
            origSumSqrs += it * it;
        }
        double origLength = Math.sqrt(origSumSqrs);
        final Map<String, Double> normWordCounts = Maps.transformValues(wordCounts, it -> it / origLength);
        log.info("Word counts (normalized): {}", normWordCounts);
        sentimentAnalyzer.normWordCounts = normWordCounts;

        return sentimentAnalyzer;
    }

    protected SentimentAnalyzer train(BayesianNetwork bn, File f, String screenName) {
        final SentimentAnalyzer analyzer = analyze(f, 100, ImmutableSet.of(screenName));

        allWords.addAll(analyzer.normWordCounts.keySet());

        // a single *user PV with > 30 dependency PVs won't be scalable, but hey, it probably works for initial demo?
        // (just to prove that it ISN'T scalable) :(
        //final ProbabilisticVariable userPv = bn.getOrCreateVariable("*user");
        //userPv.getDependencies().addAll(wordPvs);
//        final ProbabilisticState userTState = new ProbabilisticState(userPv, screenName);
//        userPv.getStates().add(userTState);
//        final ProbabilisticState userFState = new ProbabilisticState(userPv, "farhatabbaslaw");
//        userPv.getStates().add(userFState);
        // probabilities for each word = sqrt sum sqr normalized to whole training dataset (not just that user)
        // the unnormalized probabilities of user == sqrt sum sqr probabilities of words for THAT user

        for (final Map.Entry<String, Double> entry : analyzer.normWordCounts.entrySet()) {
            wordNormLengthByScreenName.put(screenName + "/" + entry.getKey(), entry.getValue());
        }

        return analyzer;
    }

    protected void train2(BayesianNetwork bn, Set<String> screenNames) {
        screenNamePv = new ProbabilisticVariable("@");
        final Map<String, ProbabilisticState> screenNameMap = screenNames.stream().map(it -> new ProbabilisticState(screenNamePv, it))
                .collect(Collectors.toMap(ProbabilisticState::getName, it -> it));
        screenNamePv.getStates().addAll(screenNameMap.values());
        // all screen names get equal probability
        for (ProbabilisticState screenNameState : screenNameMap.values()) {
            screenNamePv.getProbabilities().put(ImmutableList.of(screenNameState), 1.0 / screenNameMap.size());
        }

        wordPvs = new LinkedHashMap<>();
        for (String word : allWords) {
            final ProbabilisticVariable pv = bn.getOrCreateVariable(word); // always a new PV in this case
            wordPvs.put(word, pv);
            pv.getDependencies().add(screenNamePv);
            final ProbabilisticState fState = new ProbabilisticState(pv, ProbabilisticState.FALSE);
            pv.getStates().add(fState);
            final ProbabilisticState tState = new ProbabilisticState(pv, ProbabilisticState.TRUE);
            pv.getStates().add(tState);
            double sumWordLength = 0.0;
            for (String screenName : screenNames) {
                final Double wordNormLength = wordNormLengthByScreenName.get(screenName + "/" + word);
                if (wordNormLength != null) {
                    sumWordLength += wordNormLength;
                    pv.getProbabilities().put(ImmutableList.of(screenNameMap.get(screenName), fState), 1.0 - wordNormLength);
                    pv.getProbabilities().put(ImmutableList.of(screenNameMap.get(screenName), tState), wordNormLength);
                } else {
                    pv.getProbabilities().put(ImmutableList.of(screenNameMap.get(screenName), fState), 1.0);
                    pv.getProbabilities().put(ImmutableList.of(screenNameMap.get(screenName), tState), 0.0);
                }
            }
            if (sumWordLength <= 0.0) { // a word MUST be used by at least one
                throw new RuntimeException("Word '" + word + "' must be used by at least one screenName from " + screenNames);
            }
//            if (pv.getStates().isEmpty()) {
//                // first usage
//                final ProbabilisticState fState = new ProbabilisticState(pv, ProbabilisticState.FALSE);
//                pv.getStates().add(fState);
//                final ProbabilisticState tState = new ProbabilisticState(pv, ProbabilisticState.TRUE);
//                pv.getStates().add(tState);
//                pv.getProbabilities().put(ImmutableList.of(fState), 1.0 - entry.getValue());
//                pv.getProbabilities().put(ImmutableList.of(tState), entry.getValue());
//            } else {
//                // after first usage
//                throw new RuntimeException("Cannot handle second usage");
//            }
        }

        //final List<String> screenNames = ImmutableList.of("dakwatuna", "farhatabbaslaw");

//        // initialize permutation for probabilityKey
//        final ArrayList<ProbabilisticState> wordsKey = new ArrayList<>();
//        for (ProbabilisticVariable wordPv : wordPvs) { // next keys are dependencies
//            wordsKey.add(wordPv.getStates().get(0));
//        }
//        //probabilityKey.add(userTState); // LAST key is the variable itself
//
//        boolean hasNext = true;
//        while (hasNext) {
//            double totProb = 0.0;
//            for (final ProbabilisticState upScreenName : userPv.getStates()) {
//                final ImmutableList<ProbabilisticState> realKey = ImmutableList.copyOf(
//                        Iterables.concat(wordsKey, ImmutableList.of(upScreenName)));
//                final double prob;
//                if (screenName.equals(upScreenName.getName())) {
//                    //log.info("dakwatuna!!");
//                    double sumsqrprob = 0.0;
//                    for (ProbabilisticState wordState : wordsKey) {
//                        if (ProbabilisticState.TRUE.equals(wordState.getName())) {
//                            sumsqrprob += Math.pow(Iterables.get(wordState.getVariable().getProbabilities().values(), 1), 2);
//
//                        }
//                    }
//                    prob = Math.sqrt(sumsqrprob);
//                    log.info("sumsqrprob {}: {} -> {}", realKey, sumsqrprob, prob);
//                } else {
//                    prob = 0.0;
//                }
//                //log.info("Key: {} {}", copiedKey.hashCode(), copiedKey);
//                userPv.getProbabilities().put(realKey, prob);
//                totProb += prob;
//            }
//            // normalize
//            for (final ProbabilisticState upScreenName : userPv.getStates()) {
//                final ImmutableList<ProbabilisticState> realKey = ImmutableList.copyOf(
//                        Iterables.concat(wordsKey, ImmutableList.of(upScreenName)));
//                double prob = userPv.getProbabilities().get(realKey);
//                if (totProb > 0) {
//                    // if valid totProb, then normalize
//                    userPv.getProbabilities().put(realKey, prob / totProb);
//                } else {
//                    // otherwise, give equal weight
//                    userPv.getProbabilities().put(realKey, 1.0 / userPv.getStates().size());
//                }
//            }
//
//            // next probability
//            hasNext = nextProbabilityKey(wordsKey, wordsKey.size() - 1);
//        }

    }

    protected void testClassify(BayesianNetwork bn, File f, Set<String> screenNames,
                                String correctScreenName) {
        final SentimentAnalyzer testAnalyzer = train(bn, f, correctScreenName);
        int corrects = 0;
        int incorrects = 0;
        final Map<String, Double> sums = new LinkedHashMap<>();
        final Map<String, Integer> counts = new LinkedHashMap<>();
        screenNames.forEach(it -> {
            sums.put(it, 0.0);
            counts.put(it, 0);
        });
        for (List<String> words : testAnalyzer.words.values()) {
            for (String word : words) {
                final ProbabilisticVariable wordPv = wordPvs.get(word);
                if (wordPv != null) {
                    final Map<String, Double> ret = propagate(screenNamePv, wordPv.getStates().get(1));
                    screenNames.forEach(it -> {
                        sums.put(it, sums.get(it) + ret.get(it));
                        counts.put(it, counts.get(it) + 1);
                    });
                }
            }

            final Map<String, Double> probs = new LinkedHashMap<>();
            screenNames.forEach(it -> {
                probs.put(it, sums.get(it) / counts.get(it));
            });

            double bestProb = 0.0;
            String bestScreenName = null;
            for (String screenName : screenNames) {
                if (bestScreenName == null || probs.get(screenName) > bestProb) {
                    bestProb = probs.get(screenName);
                    bestScreenName = screenName;
                }
            }
            if (correctScreenName.equals(bestScreenName)) {
                log.info("CORRECT {} -> {} : {}",
                        words.stream().collect(Collectors.joining(" ")), probs);
                corrects++;
            } else {
                log.info("INCORRECT {} -> {} : {}",
                        words.stream().collect(Collectors.joining(" ")), probs);
                incorrects++;
            }
        }
        log.info("Correct = {}, Incorrect = {}, Total {} -> {}% accuracy", corrects, incorrects, testAnalyzer.words.size(),
                corrects * 100.0 / testAnalyzer.words.size());
    }

    @Override
    public void run(String... args) throws Exception {
        final BayesianNetwork bn = new BayesianNetwork();
        final SentimentAnalyzer dakwatunaTrain = train(bn, new File("data/tl_dakwatuna_2015-04-03_train.csv"), "dakwatuna");
        final SentimentAnalyzer farhatabbaslawTrain = train(bn, new File("data/tl_farhatabbaslaw_2015-04-03_train.csv"), "farhatabbaslaw");
        final ImmutableSet<String> screenNames = ImmutableSet.of("dakwatuna", "farhatabbaslaw");
        train2(bn, screenNames);
        log.info("BN: {}", bn.toStringComplete());

//        testClassify(bn, new File("data/tl_dakwatuna_2015-04-03_test.csv"), screenNames, "dakwatuna");
       testClassify(bn, new File("data/tl_farhatabbaslaw_2015-04-03_test.csv"), screenNames, "farhatabbaslaw");

//        propagate(screenNamePv, wordPvs.get("nasional").getStates().get(0));
//        propagate(screenNamePv, wordPvs.get("nasional").getStates().get(1));
//
//        propagate2(screenNamePv, ImmutableList.of(wordPvs.get("nasional").getStates().get(0), wordPvs.get("asia").getStates().get(0)) );
//        propagate2(screenNamePv, ImmutableList.of(wordPvs.get("nasional").getStates().get(1), wordPvs.get("asia").getStates().get(1)));

    }

    /**
     * Recursive permute.
     * @param key
     * @param permuteIdx permute from last: key.size() - 1
     * @return
     */
// permuteIdx:
    protected boolean nextProbabilityKey(ArrayList<ProbabilisticState> key, int permuteIdx) {
        final ProbabilisticState state = key.get(permuteIdx);
        final ProbabilisticVariable variable = state.getVariable();
        final int curStateIdx = variable.getStates().indexOf(state);
        if (curStateIdx < variable.getStates().size() - 1) {
            // we can just go to next state
            final int nextStateIdx = curStateIdx + 1;
            key.remove(permuteIdx);
            key.add(permuteIdx, variable.getStates().get(nextStateIdx));
            return true;
        } else if (permuteIdx > 0) {
            // oops! need to up previous segment
            boolean success = nextProbabilityKey(key, permuteIdx - 1);
            if (!success) {
                return success;
            }
            key.remove(permuteIdx);
            key.add(permuteIdx, variable.getStates().get(0));
            return true;
        } else {
            // oops! that was the final state!
            return false;
        }
    }

    /**
     * For Pearl's propagation algorithm and formal formula, see
     *
     * http://www.cse.unsw.edu.au/~cs9417ml/Bayes/Pages/PearlPropagation.html
     * http://en.wikipedia.org/wiki/Belief_propagation
     *
     * Paling gampang, langsung di:
     * The model can answer questions like "What is the probability that it is raining, given the grass is wet?" by using the conditional probability formula and summing over all nuisance variables:
     * \mathrm P(\mathit{R}=T \mid \mathit{G}=T) =\frac{ \mathrm P(\mathit{G}=T,\mathit{R}=T) } { \mathrm P(\mathit{G}=T) } =\frac{ \sum_{\mathit{S} \in \{T, F\}}\mathrm P(\mathit{G}=T,\mathit{S},\mathit{R}=T) } { \sum_{\mathit{S}, \mathit{R} \in \{T, F\}} \mathrm P(\mathit{G}=T,\mathit{S},\mathit{R}) }
     *
     * Prior:
     * P(nasional|dakwatuna) = 0.6
     * P(user=dakwatuna) = 0.5 (dari 2)
     *
     * Scenario 1:
     * Evidences: P(nasional) => 1.
     * P(user=dakwatuna | nasional) ? = P(user=dakwatuna, nasional) / P(nasional)
     * P(user=farhatabbaslaw | nasional) ?
     *
     * Scenario 2:
     * Evidences: P(nasional) => 1, P(asia) => 1.
     * P(user=dakwatuna | nasional, asia) ? = P(user=dakwatuna, nasional, asia) / P(nasional,asia)
     *
     * where
     * P(user=dakwatuna, nasional, asia) = P(user=dakwatuna) * P(nasional) * P(nasional)
     *
     * P(user=farhatabbaslaw | nasional) ?
     *
     * From http://en.wikipedia.org/wiki/Bayes%27_theorem :
     * P(nasional) = P(nasional | user=dakwatuna) P(user=dakwatuna) + P(nasional | user=farhatabbaslaw) P(user=farhatabbaslaw)
     */
    protected Map<String, Double> propagate(ProbabilisticVariable screenNamePv, ProbabilisticState wordState) {
        final LinkedHashMap<String, Double> ppgt = new LinkedHashMap<>();
        double pWord = wordState.getVariable().getStateProbability(wordState);
        for (ProbabilisticState screenNameState : screenNamePv.getStates()) {
            double pScreenName = screenNamePv.getStateProbability(screenNameState);
            double pScreenName_word = wordState.getVariable().getProbabilities().get(ImmutableList.of(screenNameState, wordState));
            double pScreenName_given_word = (pScreenName_word * pScreenName) / pWord;
            log.debug("P({}) = {}. P({}, {}) = {}. P({}) = {}. --> P({} | {}) = {}",
                    screenNameState, pScreenName,
                    screenNameState, wordState, pScreenName_word,
                    wordState, pWord,
                    screenNameState, wordState, pScreenName_given_word);
            ppgt.put(screenNameState.getName(), pScreenName_given_word);
        }
        return ppgt;
    }

    /**
     * Scenario 2:
     * Evidences: P(nasional) => 1, P(asia) => 1.
     * P(user=dakwatuna | nasional, asia) ? = P(user=dakwatuna, nasional, asia) / P(nasional, asia)
     * or?
     * P(user=dakwatuna | nasional, asia) ? = P(nasional, asia | user=dakwatuna) P(user=dakwatuna) / P(nasional, asia)
     * since P(nasional, asia) = 0.064 and P(user=*) = 0.5
     * then P(nasional, asia | user=*) must be <= 0.128
     * and P(nasional, asia | user=dakwatuna) + P(nasional, asia | user=farhatabbas) must be == 0.128
     *
     * Masalahnya, P(nasional) dan P(asia) tidak bisa langsung diambil, karena hasilnya adalah
     * P(nasional=T|nasional) dan P(nasional|F=nasional). jadi kemungkinannya:
     * 1. dibagi jumlah word (misal nasional + asia, maka dibagi 2) <-- koq ga ngefek :(
     * 2. pake root sum squared
     *
     * where [note that nasional and asia is independent, but both depend on screenName]
     * P(user=dakwatuna, nasional, asia) = P(user=dakwatuna) *
     *   P(user=dakwatuna) * P(nasional | user=dakwatuna) *
     *   P(user=dakwatuna) * P(asia | user=dakwatuna)
     * where
     * P(asia | user=dakwatuna, nasional) = P(user=dakwatuna, nasional | asia) P(nasional) / P(user=dakwatuna, nasional)
     *
     * P(user=farhatabbaslaw | nasional) ?
     *
     * From http://en.wikipedia.org/wiki/Bayes%27_theorem :
     * P(nasional) = P(nasional | user=dakwatuna) P(user=dakwatuna) + P(nasional | user=farhatabbaslaw) P(user=farhatabbaslaw)
     *
     * @param screenNamePv
     * @param wordStates
     */
    protected void propagate2(ProbabilisticVariable screenNamePv, List<ProbabilisticState> wordStates) {
        log.info("For {} word states {} :", wordStates.size(), wordStates);
        final int wordStateCount = wordStates.size();
        //double pWord = wordState.getVariable().getStateProbability(wordState);
        for (ProbabilisticState screenNameState : screenNamePv.getStates()) {
            double pScreenName = screenNamePv.getStateProbability(screenNameState);
            double mulprobs = 1.0;
            double divisorprobs = 1.0;
            String mps1 = "";
            String mps2 = "";
            String dps1 = "";
            String dps2 = "";
            for (ProbabilisticState wordState : wordStates) {
                double pWord_given_screenName = wordState.getVariable().getProbabilities().get(
                        ImmutableList.of(screenNameState, wordState)) / wordStateCount;
                mulprobs *= pWord_given_screenName;
                if (!mps1.isEmpty()) {
                    mps1 += " * ";
                    mps2 += " * ";
                }
                mps1 += "P(" + wordState + " | " + screenNameState + ")/" + wordStateCount;
                mps2 += wordState.getVariable().getProbabilities().get(
                        ImmutableList.of(screenNameState, wordState)) + "/" + wordStateCount;
                divisorprobs *= wordState.getVariable().getStateProbability(wordState) / wordStateCount;
                if (!dps1.isEmpty()) {
                    dps1 += " * ";
                    dps2 += " * ";
                }
                dps1 += "P(" + wordState + ")/" + wordStateCount;
                dps2 += wordState.getVariable().getStateProbability(wordState) + "/" + wordStateCount;
            }
            log.info("{} = {} = {}", mps1, mps2, mulprobs);
            log.info("P{} = {} = {} = {}", wordStates, dps1, dps2, divisorprobs);
            // mulprobs is wrong, because P(dakwatuna, nasional, asia) + P(farhatabbaslaw, nasional, asia) must == P(nasional, asia)
            double pScreenName_given_word = mulprobs * pScreenName / divisorprobs;

            //double pScreenName_given_word = (pScreenName_word * pScreenName) / pWord;
//            log.info("P({}) = {}. P({}, {}) = {}. P({}) = {}. --> P({} | {}) = {}",
//                    screenNameState, pScreenName,
//                    screenNameState, wordState, pScreenName_word,
//                    wordState, pWord,
//                    screenNameState, wordState, pScreenName_given_word);
            log.info("P({}) = {}. P({} | {}) = {}. P({}) = {}. --> P({} | {}) = {} * {} / {} = {}",
                    screenNameState, pScreenName,
                    wordStates, screenNameState, mulprobs,
                    wordStates, divisorprobs,
                    screenNameState, wordStates, mulprobs, pScreenName, divisorprobs, pScreenName_given_word);
        }
    }

}
