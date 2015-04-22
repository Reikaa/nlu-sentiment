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
        final ProbabilisticVariable screenNamePv = new ProbabilisticVariable("*screenName");
        final Map<String, ProbabilisticState> screenNameMap = screenNames.stream().map(it -> new ProbabilisticState(screenNamePv, it))
                .collect(Collectors.toMap(ProbabilisticState::getName, it -> it));
        screenNamePv.getStates().addAll(screenNameMap.values());
        // all screen names get equal probability
        for (ProbabilisticState screenNameState : screenNameMap.values()) {
            screenNamePv.getProbabilities().put(ImmutableList.of(screenNameState), 1.0 / screenNameMap.size());
        }

        final Map<String, ProbabilisticVariable> wordPvs = new LinkedHashMap<>();
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

    @Override
    public void run(String... args) throws Exception {
        final BayesianNetwork bn = new BayesianNetwork();
        final SentimentAnalyzer dakwatunaTrain = train(bn, new File("data/tl_dakwatuna_2015-04-03_train.csv"), "dakwatuna");
        final SentimentAnalyzer farhatabbaslawTrain = train(bn, new File("data/tl_farhatabbaslaw_2015-04-03_train.csv"), "farhatabbaslaw");
        train2(bn, ImmutableSet.of("dakwatuna", "farhatabbaslaw"));
        log.info("BN: {}", bn.toStringComplete());
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

}
