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
 * Common text mining functionality.
 * Created by ceefour on 12/04/2015.
 */
@Component
public class SentimentAnalyzer {

    private static final Logger log = LoggerFactory.getLogger(SentimentAnalyzer.class);
    /**
     * Indonesian stop words.
     */
    public static final Set<String> STOP_WORDS_ID = ImmutableSet.of(
            "di", "ke", "ini", "dengan", "untuk", "yang", "tak", "tidak", "gak",
            "dari", "dan", "atau", "bisa", "kita", "ada", "itu",
            "akan", "jadi", "menjadi", "tetap", "per", "bagi", "saat",
            "tapi", "bukan", "adalah", "pula", "aja", "saja",
            "kalo", "kalau", "karena", "pada", "kepada", "terhadap",
            "amp", // &amp;
            "rt" // RT:
    );
    /**
     * key: canonical, value: aliases
     */
    public static final Multimap<String, String> CANONICAL_WORDS;

    static {
        // Define contents of CANONICAL_WORDS
        final ImmutableMultimap.Builder<String, String> mmb = ImmutableMultimap.builder();
        mmb.putAll("yang", "yg", "yng");
        mmb.putAll("dengan", "dg", "dgn");
        mmb.putAll("saya", "sy");
        mmb.putAll("punya", "pny");
        mmb.putAll("ya", "iya");
        mmb.putAll("tidak", "tak", "tdk");
        mmb.putAll("jangan", "jgn", "jngn");
        mmb.putAll("jika", "jika", "bila");
        mmb.putAll("sudah", "udah", "sdh", "dah", "telah", "tlh");
        mmb.putAll("hanya", "hny");
        mmb.putAll("banyak", "byk", "bnyk");
        mmb.putAll("juga", "jg");
        mmb.putAll("mereka", "mrk", "mereka");
        mmb.putAll("gue", "gw", "gwe", "gua", "gwa");
        mmb.putAll("sebagai", "sbg", "sbgai");
        mmb.putAll("silaturahim", "silaturrahim", "silaturahmi", "silaturrahmi");
        mmb.putAll("shalat", "sholat", "salat", "solat");
        mmb.putAll("harus", "hrs");
        mmb.putAll("oleh", "olh");
        mmb.putAll("tentang", "ttg", "tntg");
        mmb.putAll("dalam", "dlm");
        mmb.putAll("banget", "bngt", "bgt", "bingit", "bingits");
        CANONICAL_WORDS = mmb.build();
    }

    /**
     * Normalized word counts. Key=word. Value=normalized word count.
     */
    Map<String, Double> normWordCounts;
    /**
     * Header names from the CSV file.
     */
    private String[] headerNames;
    /**
     * Raw data of CSV rows.
     */
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

    /**
     * Read CSV file {@code f} and put its contents into {@link #rows},
     * {@link #texts}, and {@link #origTexts}.
     * @param f
     */
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

    /**
     * Lower case all texts.
     */
    public void lowerCaseAll() {
        texts = Maps.transformValues(texts, String::toLowerCase);
    }

    /**
     * Remove http(s) links from texts.
     */
    public void removeLinks() {
        texts = Maps.transformValues(texts, it -> it.replaceAll("http(s?):\\/\\/(\\S+)", " "));
    }

    /**
     * Remove punctuation symbols from texts.
     */
    public void removePunctuation() {
        texts = Maps.transformValues(texts, it -> it.replaceAll("[^a-zA-Z0-9]+", " "));
    }

    /**
     * Remove numbers from texts.
     */
    public void removeNumbers() {
        texts = Maps.transformValues(texts, it -> it.replaceAll("[0-9]+", ""));
    }

    /**
     * Canonicalize different word forms using {@link #CANONICAL_WORDS}.
     */
    public void canonicalizeWords() {
        log.info("Canonicalize {} words for {} texts: {}", CANONICAL_WORDS.size(), texts.size(), CANONICAL_WORDS);
        CANONICAL_WORDS.entries().forEach(entry ->
                        texts = Maps.transformValues(texts, it -> it.replaceAll("(\\W|^)" + Pattern.quote(entry.getValue()) + "(\\W|$)", "\\1" + entry.getKey() + "\\2"))
        );
    }

    /**
     * Remove stop words using {@link #STOP_WORDS_ID} and {@code additions}.
     * @param additions
     */
    public void removeStopWords(String... additions) {
        final Sets.SetView<String> stopWords = Sets.union(STOP_WORDS_ID, ImmutableSet.copyOf(additions));
        log.info("Removing {} stop words for {} texts: {}", stopWords.size(), texts.size(), stopWords);
        stopWords.forEach(stopWord ->
            texts = Maps.transformValues(texts, it -> it.replaceAll("(\\W|^)" + Pattern.quote(stopWord) + "(\\W|$)", "\\1\\2"))
        );
    }

    /**
     * Split texts into {@link #words}.
     */
    public void splitWords() {
        Splitter whitespace = Splitter.on(Pattern.compile("\\s+")).omitEmptyStrings().trimResults();
        words = Maps.transformValues(texts, it -> whitespace.splitToList(it));
    }

}
