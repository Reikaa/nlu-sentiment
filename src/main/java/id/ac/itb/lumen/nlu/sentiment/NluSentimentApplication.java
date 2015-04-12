package id.ac.itb.lumen.nlu.sentiment;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.annotation.Profile;

@SpringBootApplication
@Profile("nlu-sentiment")
public class NluSentimentApplication implements CommandLineRunner {

    private static Logger log = LoggerFactory.getLogger(NluSentimentApplication.class);
    
    public static void main(String[] args) {
        new SpringApplicationBuilder(NluSentimentApplication.class)
                .profiles("nlu-sentiment")
                .run(args);
    }

    @Override
    public void run(String... args) throws Exception {
        log.info("Hai");
    }
}
