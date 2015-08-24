using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ArtificialNeuralNetworkLibrary
{
  public abstract class Layer
  {
    protected int inputDim;
    protected int inputDepthDim;
    protected string activationFunction;

    /// <summary>
    /// Abstracts layer constructor
    /// </summary>
    /// <param name="inputDim"></param>
    /// <param name="inputDepthDim"></param>
    /// <param name="activationFunction"></param>
    protected Layer(int inputDim, int inputDepthDim, string activationFunction)
    {
      this.inputDim = inputDim;
      this.inputDepthDim = inputDepthDim;
      this.activationFunction = activationFunction.ToLower();
    }
  }
}