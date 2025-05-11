package com.example.mlalgorithm.algorithm.markov;

import lombok.Data;

/**
 * 天气预测场景
 * 使用马尔科夫链预测天气变化
 */
@Data
public class WeatherPrediction {
    private MarkovChain markovChain;
    
    // 天气状态定义
    public static final int SUNNY = 0;    // 晴天
    public static final int CLOUDY = 1;   // 多云
    public static final int RAINY = 2;    // 雨天
    
    public WeatherPrediction() {
        // 创建3个状态的马尔科夫链
        markovChain = new MarkovChain(3);
        
        // 设置转移概率
        // 晴天到晴天的概率
        markovChain.setTransitionProbability(SUNNY, SUNNY, 0.7);
        // 晴天到多云的概率
        markovChain.setTransitionProbability(SUNNY, CLOUDY, 0.2);
        // 晴天到雨天的概率
        markovChain.setTransitionProbability(SUNNY, RAINY, 0.1);
        
        // 多云到晴天的概率
        markovChain.setTransitionProbability(CLOUDY, SUNNY, 0.3);
        // 多云到多云的概率
        markovChain.setTransitionProbability(CLOUDY, CLOUDY, 0.4);
        // 多云到雨天的概率
        markovChain.setTransitionProbability(CLOUDY, RAINY, 0.3);
        
        // 雨天到晴天的概率
        markovChain.setTransitionProbability(RAINY, SUNNY, 0.2);
        // 雨天到多云的概率
        markovChain.setTransitionProbability(RAINY, CLOUDY, 0.3);
        // 雨天到雨天的概率
        markovChain.setTransitionProbability(RAINY, RAINY, 0.5);
        
        // 设置初始状态概率
        markovChain.setInitialStateProbability(SUNNY, 0.4);
        markovChain.setInitialStateProbability(CLOUDY, 0.3);
        markovChain.setInitialStateProbability(RAINY, 0.3);
    }
    
    /**
     * 预测明天的天气
     * @param todayWeather 今天的天气状态
     * @return 明天的天气状态
     */
    public int predictTomorrowWeather(int todayWeather) {
        return markovChain.predictNextState(todayWeather);
    }
    
    /**
     * 获取天气状态描述
     * @param weatherState 天气状态
     * @return 天气描述
     */
    public static String getWeatherDescription(int weatherState) {
        switch (weatherState) {
            case SUNNY:
                return "晴天";
            case CLOUDY:
                return "多云";
            case RAINY:
                return "雨天";
            default:
                return "未知";
        }
    }
} 