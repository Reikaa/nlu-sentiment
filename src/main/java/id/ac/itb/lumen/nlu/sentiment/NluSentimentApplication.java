package id.ac.itb.lumen.nlu.sentiment;

import com.google.common.collect.Iterables;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.annotation.Profile;

import java.io.File;
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

    @Autowired
    private SentimentAnalyzer sentimentAnalyzer;

    @Override
    public void run(String... args) throws Exception {
        sentimentAnalyzer.readCsv(new File("data/tl_dakwatuna_2015-04-03_tagged.csv"));
        sentimentAnalyzer.lowerCaseAll();
        sentimentAnalyzer.removeLinks();
        sentimentAnalyzer.removePunctuation();
        sentimentAnalyzer.removeNumbers();
        log.info("Preprocessed text: {}", sentimentAnalyzer.texts.entrySet().stream().limit(10)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
        sentimentAnalyzer.splitWords();;
        log.info("Words: {}", sentimentAnalyzer.words.entrySet().stream().limit(10)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
    }
}
