using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ArtificialNeuralNetworkLibrary
{
  /// <summary>
  /// A convolutional connected layer class. 
  /// 
  /// Indices:
  /// Neurons in this layer:
  /// i - denotes row-position of neuron 
  /// j - denotes column-postion of neuron 
  /// f - denotes z-positon i.e filter or feature map of neuron
  /// 
  /// Input Neurons in previous layer:
  /// ii - denotes row-position of neuron 
  /// jj - denotes column-postion of neuron 
  /// ff - denotes z-positon i.e input channel or filter/feature map of neuron
  /// 
  /// Weights:
  /// f - denotes feature map index in this layer
  /// a - denotes row-position of filter
  /// b - denotes column-postion of filter
  /// ff - denotes z-positon i.e input channel or filter/feature map of neuron
  /// </summary>
  public class ConvolutionalLayer : Layer
  {
    // Layer parameters:
    #region
    private int nrOfFilters;
    private int filterDim;
    private int filterStride;

    private int featureMapDim;
    private int fanIn;
    #endregion

    /// <summary>
    /// Constructor which initializes the paramters for the layer
    /// </summary>
    /// <param name="inputDim"></param>
    /// <param name="inputDepthDim"></param>
    /// <param name="activationFunction"></param>
    /// <param name="nrOfFilters"></param>
    /// <param name="filterDim"></param>
    /// <param name="filterStride"></param>
    public ConvolutionalLayer(int inputDim, int inputDepthDim, string activationFunction,
      int nrOfFilters, int filterDim, int filterStride) : base(inputDim, inputDepthDim, activationFunction)
    {
      this.nrOfFilters = nrOfFilters;
      this.filterDim = filterDim;
      this.filterStride = filterStride;

      if ( ((this.inputDim - this.filterDim) / filterStride) % 1 != 0) {
        throw new System.ArgumentException("Invalid input arguments i.e. (inputSize - filterSize)/Stride must be an integer.");
      }

      this.featureMapDim = (this.inputDim - this.filterDim) / filterStride + 1;
      this.fanIn = this.filterDim * this.filterDim * this.inputDepthDim;
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

      for (int f = 0; f < this.nrOfFilters; ++f) {
        biases[f] = (rnd.NextDouble() - 0.5) * 2 / Math.Sqrt(this.fanIn);
        for (int a = 0; a < this.filterDim; ++a) {
          for (int b = 0; b < this.filterDim; ++b) {
            for (int ff = 0; ff < this.inputDepthDim; ++ff) {
              weights[f][a][b][ff] = (rnd.NextDouble() - 0.5) * 2 / Math.Sqrt(this.fanIn);
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
    public void ForwardPropagate(double[][][] previousNeurons, double[][][] neurons, double[][][][] weights, double[] biases)
    {
      if (this.activationFunction == "relu") { // Rectified linear function
        #region
        Parallel.For(0, this.nrOfFilters, f => {
          for (int i = 0; i < this.featureMapDim; ++i) {
            for (int j = 0; j < this.featureMapDim; ++j) {

              // Top left corner ind of the convolving filter
              int ii = i * this.filterStride;
              int jj = j * this.filterStride;

              double preActivation = biases[f];
              for (int a = 0; a < this.filterDim; ++a) {
                for (int b = 0; b < this.filterDim; ++b) {
                  for (int ff = 0; ff < this.inputDepthDim; ++ff) {
                    preActivation += weights[f][a][b][ff] * previousNeurons[ii + a][jj + b][ff];
                  }
                }
              }
              neurons[i][j][f] = preActivation > 0 ? preActivation : 0;

            }
          }
        });
        #endregion
      } else if (this.activationFunction == "sigmoid") { // Sigmoid activation function
        #region
        Parallel.For(0, this.nrOfFilters, f => {
          for (int i = 0; i < this.featureMapDim; ++i) {
            for (int j = 0; j < this.featureMapDim; ++j) {

              int ii = i * this.filterStride;
              int jj = j * this.filterStride;

              double preActivation = biases[f];
              for (int a = 0; a < this.filterDim; ++a) {
                for (int b = 0; b < this.filterDim; ++b) {
                  for (int ff = 0; ff < this.inputDepthDim; ++ff) {
                    preActivation += weights[f][a][b][ff] * previousNeurons[ii + a][jj + b][ff];
                  }
                }
              }
              neurons[i][j][f] = 1 / (1 + Math.Exp(-preActivation));

            }
          }
        });
        #endregion
      } else { // Tanh activation function
        #region
        Parallel.For(0, this.nrOfFilters, f => {
          for (int i = 0; i < this.featureMapDim; ++i) {
            for (int j = 0; j < this.featureMapDim; ++j) {

              int ii = i * this.filterStride;
              int jj = j * this.filterStride;

              double preActivation = biases[f];
              for (int a = 0; a < this.filterDim; ++a) {
                for (int b = 0; b < this.filterDim; ++b) {
                  for (int ff = 0; ff < this.inputDepthDim; ++ff) {
                    preActivation += weights[f][a][b][ff] * previousNeurons[ii + a][jj + b][ff];
                  }
                }
              }
              neurons[i][j][f] = Math.Tanh(preActivation);

            }
          }
        });
        #endregion
      }

    }


    /// <summary>
    /// Calculating the weight and bias gradients based on the deltas from this layers backpropagate
    /// </summary>
    /// <param name="initialDeltas"></param>
    /// <param name="previousNeurons"></param>
    /// <param name="weightGradients"></param>
    /// <param name="biasGradients"></param>
    public void AddWeightAndBiasGradients(double[][][] propagatedDeltas, double[][][] previousNeurons, double[][][][] weightGradients, double[] biasGradients)
    {
      Parallel.For(0, this.nrOfFilters, f => { // looping over featuremaps
        // Calculating bias gradients
        #region
        for (int i = 0; i < this.featureMapDim; ++i) {  // looping over neurons in this layer
          for (int j = 0; j < this.featureMapDim; ++j) {
            biasGradients[f] += propagatedDeltas[i][j][f];
          }
        }
        #endregion

        // Calculating weight gradients
        #region
        for (int a = 0; a < this.filterDim; ++a) {  // looping over filter in previous layer
          for (int b = 0; b < this.filterDim; ++b) {
            for (int ff = 0; ff < this.inputDepthDim; ++ff) {

              for (int i = 0; i < this.featureMapDim; ++i) {  // looping over neurons in this layer
                for (int j = 0; j < this.featureMapDim; ++j) {

                  // Top left corner ind of the convolving filter
                  int ii = i * this.filterStride;
                  int jj = j * this.filterStride;

                  weightGradients[f][a][b][ff] += propagatedDeltas[i][j][f] * previousNeurons[ii + a][jj + b][ff];
                }
              }

            }
          }
        }
        #endregion
      });

    }

    
    /// <summary>
    /// Returns the activation funciton of this layer
    /// </summary>
    /// <returns></returns>
    public string GetActivationFunction()
    {
      return this.activationFunction;
    }

  }
}
