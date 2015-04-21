package id.ac.itb.lumen.nlu.sentiment;

import com.google.common.collect.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.annotation.Profile;

import javax.inject.Inject;
import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;
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

    @Inject
    private SentimentAnalyzer sentimentAnalyzer;

    @Override
    public void run(String... args) throws Exception {
        sentimentAnalyzer.readCsv(new File("data/tl_dakwatuna_2015-04-03_tagged.csv"));
        sentimentAnalyzer.lowerCaseAll();
        sentimentAnalyzer.removeLinks();
        sentimentAnalyzer.removePunctuation();
        sentimentAnalyzer.removeNumbers();
        sentimentAnalyzer.removeStopWords("dakwatuna");
        log.info("Preprocessed text: {}", sentimentAnalyzer.texts.entrySet().stream().limit(10)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
        sentimentAnalyzer.splitWords();
        log.info("Words: {}", sentimentAnalyzer.words.entrySet().stream().limit(10)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));

        final ImmutableMultiset<String> wordMultiset = Multisets.copyHighestCountFirst(HashMultiset.create(
                sentimentAnalyzer.words.values().stream().flatMap(it -> it.stream()).collect(Collectors.toList())) );
        final Map<String, Integer> wordCounts = new LinkedHashMap<>();
        // only the 100 most used words
        wordMultiset.elementSet().stream().limit(100).forEach( it -> wordCounts.put(it, wordMultiset.count(it)) );
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
    }
}
