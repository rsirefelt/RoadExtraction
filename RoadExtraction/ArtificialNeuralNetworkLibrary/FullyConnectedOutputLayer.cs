using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ArtificialNeuralNetworkLibrary
{
  /// <summary>
  /// A fully connected output layer class. 
  /// It has two primary containers:
  /// 
  /// Indices:
  /// i - denotes x-position of neuron in previous layer
  /// j - denotes y-postion of neuron in previous layer
  /// f - denotes z-positon (filter or input channel) of neuron in previous layer
  /// k - denotes neuron index in this layer
  /// </summary>
  public class FullyConnectedOutputLayer : Layer
  {
    // Layer paramters:
    #region
    private int outputDim;
    private int nrOfOutputs;
    private string lossFunction;

    private int fanIn;
    double[] initialDeltas;  // deltas(k) i.e. 1D, but represents 2D output
    double[][][] propagatedDeltas;
    #endregion

    /// <summary>
    /// Constructor which initializes the weights and neurons of the fully connected output layer
    /// </summary>
    /// <param name="inputDim"></param>
    /// <param name="inputDepthDim"></param>
    /// <param name="activationFunction"></param>
    /// <param name="outputDim"></param>
    /// <param name="lossFunction"></param>
    public FullyConnectedOutputLayer(int inputDim, int inputDepthDim, string activationFunction, int outputDim, string lossFunction)
      : base(inputDim, inputDepthDim, activationFunction)
    {
      this.outputDim = outputDim;
      this.nrOfOutputs = outputDim * outputDim;
      this.lossFunction = lossFunction.ToLower();
      
      this.fanIn = this.inputDim * this.inputDim * this.inputDepthDim;
      this.initialDeltas = MiscLibrary.Initialize.Allocate<double>(this.nrOfOutputs);
      this.propagatedDeltas = MiscLibrary.Initialize.Allocate<double>(this.inputDim, this.inputDim, this.inputDepthDim);
    }


    /// <summary>
    /// Initializes all of the weights and biases to random values.
    /// The random values for the weights depend on the fan-in to the network.
    /// </summary>
    /// <param name="weights"></param>
    /// <param name="biases"></param>
    public void InitializeWeightsAndBiases(double[][][][] weights, double[] biases)
    {
      Random rnd = new Random();
      for (int k = 0; k < this.nrOfOutputs; ++k) {
        biases[k] = (rnd.NextDouble() - 0.5) * 2 / Math.Sqrt(this.fanIn);
        for (int i = 0; i < this.inputDim; ++i) {
          for (int j = 0; j < this.inputDim; ++j) {
            for (int f = 0; f < this.inputDepthDim; ++f) {
              weights[k][i][j][f] = (rnd.NextDouble() - 0.5) * 2 / Math.Sqrt(this.fanIn);
            }
          }
        }
      }

    }


    /// <summary>
    /// Calculating the value of all neurons in the layer, i.e a forward pass through the network.
    /// This function alters the values in the second neuron parameter "neurons".
    /// </summary>
    /// <param name="previousNeurons"></param>
    /// <param name="neurons"></param>
    /// <param name="weights"></param>
    /// <param name="biases"></param>
    public void ForwardPropagate(double[][][] previousNeurons, double[] neurons, double[][][][] weights, double[] biases)
    {

      if (this.activationFunction == "sigmoid") { // Sigmoid activation function
        #region
        Parallel.For(0, this.nrOfOutputs, k => {
          double preActivation = biases[k];
          for (int i = 0; i < this.inputDim; ++i) {
            for (int j = 0; j < this.inputDim; ++j) {
              for (int f = 0; f < this.inputDepthDim; ++f) {
                preActivation += weights[k][i][j][f] * previousNeurons[i][j][f];

              }
            }
          }
          neurons[k] = 1.0 / (1.0 + Math.Exp(-preActivation));
        });
        #endregion
      } else { // Tanh activation function
        #region
        Parallel.For(0, this.nrOfOutputs, k => {
          double preActivation = biases[k];
          for (int i = 0; i < this.inputDim; ++i) {
            for (int j = 0; j < this.inputDim; ++j) {
              for (int f = 0; f < this.inputDepthDim; ++f) {
                preActivation += weights[k][i][j][f] * previousNeurons[i][j][f];
              }
            }
          }
          neurons[k] = Math.Tanh(preActivation);
        });
        #endregion
      }
    }


    /// <summary>
    /// Calculates the inital backward message that used by the backward propagate function in this layer
    /// </summary>
    /// <param name="trainData"></param>
    /// <param name="neurons"></param>
    /// <returns></returns>
    public double[] CalculateInitialDeltas(byte[] trainData, double[] neurons)
    {
      if (this.lossFunction == "crossentropy") { // Loss function: Cross entropy
        if (this.activationFunction == "sigmoid") { // Output activation function: Sigmoid
          for (int k = 0; k < this.nrOfOutputs; ++k) {
            initialDeltas[k] = -(trainData[k] - neurons[k]);
          }
        } else { // Output activation function: Tanh
          for (int k = 0; k < this.nrOfOutputs; ++k) {
            initialDeltas[k] = -(trainData[k] / neurons[k] - (1 - trainData[k]) / (1 - neurons[k])) * (1 - neurons[k] * neurons[k]);
          }
        }
      } else { // Loss function: Squared Error
        if (this.activationFunction == "sigmoid") { // Output activation function: Sigmoid
          for (int k = 0; k < this.nrOfOutputs; ++k) {
            initialDeltas[k] = -(trainData[k] - neurons[k]) * neurons[k] * (1 - neurons[k]);
          }
        } else { // Output activation function: Tanh
          for (int k = 0; k < this.nrOfOutputs; ++k) {
            initialDeltas[k] = -(trainData[k] - neurons[k]) * (1 - neurons[k] * neurons[k]);
          }
        }
      }

      return initialDeltas;
    }


    /// <summary>
    /// Calculates the backward message that is used in this layer and the previous layer.
    /// Must be followed by calculating the initial deltas.
    /// </summary>
    /// <param name="weights"></param>
    /// <param name="previousNeurons"></param>
    /// <param name="previousActivationFunction"></param>
    /// <returns></returns>
    public double[][][] BackwardPropagate(double[][][][] weights, double[][][] previousNeurons, string previousActivationFunction)
    {
      if (previousActivationFunction == "relu") { // Previous activation function: Rectified linear unit
        #region
        Parallel.For(0, this.inputDim, i => {
          for (int j = 0; j < this.inputDim; ++j) {
            for (int f = 0; f < this.inputDepthDim; ++f) {

              propagatedDeltas[i][j][f] = 0;
              if (previousNeurons[i][j][f] > 0) {
                for (int k = 0; k < this.nrOfOutputs; ++k) {
                  propagatedDeltas[i][j][f] += initialDeltas[k] * weights[k][i][j][f];
                }
              }

            }
          }
        });
        #endregion
      } else if (this.activationFunction == "sigmoid") { // Previous activation function: Sigmoid
        #region
        Parallel.For(0, this.inputDim, i => {
          for (int j = 0; j < this.inputDim; ++j) {
            for (int f = 0; f < this.inputDepthDim; ++f) {

              propagatedDeltas[i][j][f] = 0;
              for (int k = 0; k < this.nrOfOutputs; ++k) {
                propagatedDeltas[i][j][f] += initialDeltas[k] * weights[k][i][j][f];
              }
              propagatedDeltas[i][j][f] *= previousNeurons[i][j][f] * (1 - previousNeurons[i][j][f]);

            }
          }
        });
        #endregion
      } else { // Previous activation function: Tanh
        #region
        Parallel.For(0, this.inputDim, i => {
          for (int j = 0; j < this.inputDim; ++j) {
            for (int f = 0; f < this.inputDepthDim; ++f) {

              propagatedDeltas[i][j][f] = 0;
              for (int k = 0; k < propagatedDeltas.Length; ++k) {
                propagatedDeltas[i][j][f] += initialDeltas[k] * weights[k][i][j][f];
              }
              propagatedDeltas[i][j][f] *= 1 - previousNeurons[i][j][f] * previousNeurons[i][j][f];

            }
          }
        });
        #endregion
      }

      return propagatedDeltas;
    }


    /// <summary>
    /// Calculating and adds the weight and bias gradients based on the deltas from this layers backpropagate
    /// </summary>
    /// <param name="initialDeltas"></param>
    /// <param name="previousNeurons"></param>
    /// <param name="weightGradients"></param>
    /// <param name="biasGradients"></param>
    public void AddWeightAndBiasGradients(double[] initialDeltas, double[][][] previousNeurons, double[][][][] weightGradients, double[] biasGradients)
    {
      Parallel.For(0, this.nrOfOutputs, k => {

        biasGradients[k] += initialDeltas[k];

        for (int i = 0; i < this.inputDim; ++i) {
          for (int j = 0; j < this.inputDim; ++j) {
            for (int f = 0; f < this.inputDepthDim; ++f) {
              weightGradients[k][i][j][f] += initialDeltas[k] * previousNeurons[i][j][f];
            }
          }
        }
      });

    }


    /// <summary>
    /// Returns the current predicted road patch, i.e. which 
    /// road segment the current state of the output neurons would predict.
    /// </summary>
    /// <param name="neurons"></param>
    /// <param name="roadProbabilityThreshold"></param>
    /// <returns></returns>
    public byte[] GetCurrentPredictedRoadPatch(double[] neurons, double roadProbabilityThreshold)
    {
      int nrOfNetworkOutputs = neurons.Length;
      byte[] predictedRoadSegment = new byte[nrOfNetworkOutputs];

      for (int k = 0; k < nrOfNetworkOutputs; ++k) {
        predictedRoadSegment[k] = (neurons[k] > roadProbabilityThreshold) ? (byte)255 : (byte)0;
      }

      return predictedRoadSegment;
    }


    /// <summary>
    /// Returns the current predicted raw road patch, i.e. translates the current
    /// raw neurons state to pixel values between 0 - 255.
    /// </summary>
    /// <returns></returns>
    public byte[] GetCurrentPredictedRawRoadPatch(double[] neurons)
    {
      int nrOfNetworkOutputs = neurons.Length;
      byte[] predictedRawRoadSegment = new byte[nrOfNetworkOutputs];

      for (int k = 0; k < nrOfNetworkOutputs; ++k) {
        predictedRawRoadSegment[k] = (byte)Math.Round(neurons[k] * 255);
      }

      return predictedRawRoadSegment;
    }


    /////////////////////////////// Loss functions ////////////////////////////////////////////////////////


    /// <summary>
    /// Calculates the current loss for the current neuron state
    /// </summary>
    /// <param name="goldData"></param>
    /// <param name="neurons"></param>
    /// <returns></returns>
    public double GetCurrentLoss(byte[] goldData, double[] neurons)
    {
      if (goldData.Length != this.nrOfOutputs) {
        throw new System.ArgumentException("Illegal input dimesions for the train data array.");
      }

      double loss = 0;
      if (this.lossFunction == "crossentropy") { // Loss function: Cross entropy 
        for (int k = 0; k < this.nrOfOutputs; ++k) {
          loss -= goldData[k] * Math.Log(neurons[k]) + (1 - goldData[k]) * Math.Log(1 - neurons[k]);
        }
      } else { // Loss function: Squared Error
        for (int k = 0; k < this.nrOfOutputs; ++k) {
          loss += 0.5 * (goldData[k] - neurons[k]) * (goldData[k] - neurons[k]);
        }
      }

      return loss;
    }


    /// <summary>
    /// Calculates the current classification error for the current neuron state.
    /// </summary>
    /// <param name="goldPatch"></param>
    /// <param name="neurons"></param>
    /// <param name="roadProbabilityThreshold"></param>
    /// <returns></returns>
    public int GetCurrentClassificationError(byte[] goldPatch, double[] neurons, double roadProbabilityThreshold)
    {
      byte pixelClassification;
      int patchClassificationError = 0;
      byte nonRoadValue = 0;
      byte roadValue = 1;

      int nrOfNetworkOutputs = neurons.Length;
      for (int k = 0; k < nrOfNetworkOutputs; ++k) {
        pixelClassification = neurons[k] > roadProbabilityThreshold ? roadValue : nonRoadValue;
        patchClassificationError += Math.Abs(pixelClassification - goldPatch[k]);
      }

      return patchClassificationError;
    }

  }
}
