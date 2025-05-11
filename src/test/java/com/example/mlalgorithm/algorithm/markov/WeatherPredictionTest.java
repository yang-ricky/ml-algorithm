package com.example.mlalgorithm.algorithm.markov;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

class WeatherPredictionTest {
    private WeatherPrediction weatherPrediction;
    private List<Integer> testSequence;

    @BeforeEach
    void setUp() {
        weatherPrediction = new WeatherPrediction();
        testSequence = new ArrayList<>();
        
        // 从CSV文件加载测试数据
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(getClass().getResourceAsStream("/weather_sequence.csv")))) {
            // 跳过标题行
            reader.readLine();
            
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                testSequence.add(Integer.parseInt(parts[1]));
            }
        } catch (Exception e) {
            fail("Failed to load test data: " + e.getMessage());
        }
    }

    @Test
    void testPredictTomorrowWeather() {
        // 测试从晴天预测
        int nextWeather = weatherPrediction.predictTomorrowWeather(WeatherPrediction.SUNNY);
        assertTrue(nextWeather >= 0 && nextWeather <= 2, "预测的天气状态应该在有效范围内");
        
        // 测试从雨天预测
        nextWeather = weatherPrediction.predictTomorrowWeather(WeatherPrediction.RAINY);
        assertTrue(nextWeather >= 0 && nextWeather <= 2, "预测的天气状态应该在有效范围内");
    }

    @Test
    void testWeatherSequencePrediction() {
        // 使用测试序列进行预测
        for (int i = 0; i < testSequence.size() - 1; i++) {
            int currentWeather = testSequence.get(i);
            int actualNextWeather = testSequence.get(i + 1);
            int predictedNextWeather = weatherPrediction.predictTomorrowWeather(currentWeather);
            
            System.out.printf("当前天气: %s, 实际下一天天气: %s, 预测下一天天气: %s%n",
                    WeatherPrediction.getWeatherDescription(currentWeather),
                    WeatherPrediction.getWeatherDescription(actualNextWeather),
                    WeatherPrediction.getWeatherDescription(predictedNextWeather));
            
            // 由于马尔科夫链是概率性的，我们只验证预测结果在有效范围内
            assertTrue(predictedNextWeather >= 0 && predictedNextWeather <= 2,
                    "预测的天气状态应该在有效范围内");
        }
    }

    @Test
    void testWeatherDescription() {
        assertEquals("晴天", WeatherPrediction.getWeatherDescription(WeatherPrediction.SUNNY));
        assertEquals("多云", WeatherPrediction.getWeatherDescription(WeatherPrediction.CLOUDY));
        assertEquals("雨天", WeatherPrediction.getWeatherDescription(WeatherPrediction.RAINY));
        assertEquals("未知", WeatherPrediction.getWeatherDescription(999));
    }
}