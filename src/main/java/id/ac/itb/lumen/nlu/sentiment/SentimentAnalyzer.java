package id.ac.itb.lumen.nlu.sentiment;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.opencsv.CSVReader;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by ceefour on 12/04/2015.
 */
@Component
public class SentimentAnalyzer {

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

    public void splitWords() {
        Splitter whitespace = Splitter.on(Pattern.compile("\\s+")).omitEmptyStrings().trimResults();
        words = Maps.transformValues(texts, it -> whitespace.splitToList(it));
    }

}
