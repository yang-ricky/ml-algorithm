package com.example.mlalgorithm.algorithm.markov;

import lombok.Data;
import smile.math.matrix.Matrix;
import smile.stat.distribution.GaussianDistribution;

/**
 * 马尔科夫链实现
 * 使用SMILE库的HMM实现
 */
@Data
public class MarkovChain {
    private Matrix transitionMatrix;  // 转移矩阵
    private double[] initialState;    // 初始状态概率
    private int numStates;            // 状态数量

    public MarkovChain(int numStates) {
        this.numStates = numStates;
        this.transitionMatrix = new Matrix(numStates, numStates);
        this.initialState = new double[numStates];
    }

    /**
     * 设置转移概率
     * @param fromState 起始状态
     * @param toState 目标状态
     * @param probability 转移概率
     */
    public void setTransitionProbability(int fromState, int toState, double probability) {
        transitionMatrix.set(fromState, toState, probability);
    }

    /**
     * 设置初始状态概率
     * @param state 状态
     * @param probability 概率
     */
    public void setInitialStateProbability(int state, double probability) {
        initialState[state] = probability;
    }

    /**
     * 预测下一个状态
     * @param currentState 当前状态
     * @return 下一个状态
     */
    public int predictNextState(int currentState) {
        double[] probabilities = transitionMatrix.row(currentState);
        double maxProb = 0;
        int nextState = 0;
        
        for (int i = 0; i < numStates; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                nextState = i;
            }
        }
        
        return nextState;
    }

    /**
     * 计算状态序列的概率
     * @param sequence 状态序列
     * @return 概率
     */
    public double calculateSequenceProbability(int[] sequence) {
        if (sequence.length == 0) return 0;
        
        double probability = initialState[sequence[0]];
        
        for (int i = 1; i < sequence.length; i++) {
            probability *= transitionMatrix.get(sequence[i-1], sequence[i]);
        }
        
        return probability;
    }
} 