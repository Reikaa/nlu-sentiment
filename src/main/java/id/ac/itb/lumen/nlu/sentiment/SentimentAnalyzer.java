package id.ac.itb.lumen.nlu.sentiment;

import com.google.common.base.Splitter;
import com.google.common.collect.*;
import com.opencsv.CSVReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by ceefour on 12/04/2015.
 */
@Component
public class SentimentAnalyzer {

    private static Logger log = LoggerFactory.getLogger(SentimentAnalyzer.class);
    public static Set<String> STOP_WORDS_ID = ImmutableSet.of(
            "di", "ke", "ini", "dengan", "untuk", "yang", "tak", "tidak", "gak",
            "dari", "dan", "atau", "bisa", "kita", "ada", "itu",
            "akan", "jadi", "menjadi", "tetap", "per", "bagi", "saat",
            "tapi", "bukan", "adalah", "pula", "aja", "saja",
            "kalo", "kalau", "karena", "pada", "kepada", "terhadap",
            "amp" // &amp;
    );

    private String[] headerNames;
    private List<String[]> rows;
    /**
     * key=row ID. value=original text
     */
    private ImmutableMap<String, String> origTexts;
    /**
     * key=row ID. value=preprocessed text
     */
    Map<String, String> texts;
    /**
     * key=row ID. value=ordered list of words
     */
    Map<String, List<String>> words;

    public void readCsv(File f) {
        try (final CSVReader csv = new CSVReader(new FileReader(f))) {
            headerNames = csv.readNext(); // header
            rows = csv.readAll();
            texts = rows.stream().map(it -> Maps.immutableEntry(it[0], it[1]))
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
            origTexts = ImmutableMap.copyOf(texts);
        } catch (Exception e) {
            throw new RuntimeException("Cannot read " + f, e);
        }
    }

    public void lowerCaseAll() {
        texts = Maps.transformValues(texts, String::toLowerCase);
    }

    public void removeLinks() {
        texts = Maps.transformValues(texts, it -> it.replaceAll("http(s?):\\/\\/(\\S+)", " "));
    }

    public void removePunctuation() {
        texts = Maps.transformValues(texts, it -> it.replaceAll("[^a-zA-Z0-9]+", " "));
    }

    public void removeNumbers() {
        texts = Maps.transformValues(texts, it -> it.replaceAll("[0-9]+", ""));
    }

    public void removeStopWords(String... additions) {
        final Sets.SetView<String> stopWords = Sets.union(STOP_WORDS_ID, ImmutableSet.copyOf(additions));
        log.info("Removing {} stop words for {} texts: {}", stopWords.size(), texts.size(), stopWords);
        stopWords.forEach(stopWord ->
            texts = Maps.transformValues(texts, it -> it.replaceAll("(\\W|^)" + Pattern.quote(stopWord) + "(\\W|$)", ""))
        );
    }

    public void splitWords() {
        Splitter whitespace = Splitter.on(Pattern.compile("\\s+")).omitEmptyStrings().trimResults();
        words = Maps.transformValues(texts, it -> whitespace.splitToList(it));
    }

}
