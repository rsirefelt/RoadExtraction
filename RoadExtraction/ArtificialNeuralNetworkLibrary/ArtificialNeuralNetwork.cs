using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace ArtificialNeuralNetworkLibrary
{
  /// <summary>
  /// An artificial neural network class. 
  /// </summary>
  public class ArtificialNeuralNetwork
  {
    // Network parameters:
    #region
    private const byte NR_OF_PIXEL_CHANNELS = 3;
    private const int  NR_OF_LAYERS = 2;
    private const int CONV_LAYER_IND = 0;
    private const int OUTPUT_LAYER_IND = 1;

    private int networkInputDim;
    private int networkOutputDim;
    private int nrOfNetworkOutputs;
    private int outputOffset;

    private int[][] weightDim; // For Conv layer: W(f,a,b,ff): (a,b,ff) = filter position,  f = feature map index next layer, 
                               // For Output layer: W(k,i,j,f): (i,j,f) = neuron position input layer,  k = neuron index this layer.
    
    private double[][][][][] weights; // weights for the two layers
    private double[][][][][] weightChanges; // weight changes for the two layers, used for minibatch weight update
    private double[][][][][] weightGradients;

    private double[][] biases; // biases for the two layers.
    private double[][] biasChanges; // bias changes the two layers, used for minibatch bias update
    private double[][] biasGradients;

    private double[] neuronsOutputLayer; // neurons in the outputlayer
    private double[][][] neuronsConvLayer; // neurons in the convolutional layer

    private double stdDevTrainSet;
    #endregion

    ArtificialNeuralNetworkLibrary.FullyConnectedOutputLayer outputLayer;
    ArtificialNeuralNetworkLibrary.ConvolutionalLayer convolutionalLayer;

    /// <summary>
    /// Constructor which sets a bunch of network paramters
    /// </summary>
    public ArtificialNeuralNetwork(int[] weightDimForConv, int[] weightDimForOutput, int networkInputDim, int networkOutputDim)
    {
      this.weightDim = new int[NR_OF_LAYERS][];
      this.weights = new double[NR_OF_LAYERS][][][][];
      this.weightChanges = new double[NR_OF_LAYERS][][][][];
      this.weightGradients = new double[NR_OF_LAYERS][][][][];
      
      this.biases = new double[NR_OF_LAYERS][];
      this.biasChanges = new double[NR_OF_LAYERS][];
      this.biasGradients = new double[NR_OF_LAYERS][];

      this.weightDim[CONV_LAYER_IND] = weightDimForConv;
      this.weightDim[OUTPUT_LAYER_IND] = weightDimForOutput;

      this.networkInputDim = networkInputDim;
      this.networkOutputDim = networkOutputDim;
      this.nrOfNetworkOutputs = networkOutputDim * networkOutputDim;
      this.outputOffset = (this.networkInputDim - this.networkOutputDim) / 2;
    }

    // Initialize layers
    #region
    /// <summary>
    /// Constructs and initializes the weights biases and paramters for the convolution layer
    /// </summary>
    /// <param name="inputDim"></param>
    /// <param name="inputDepthDim"></param>
    /// <param name="activationFunction"></param>
    /// <param name="nrOfFilters"></param>
    /// <param name="filterDim"></param>
    /// <param name="filterStride"></param>
    public void InitializeConvolutionalLayer(int inputDim, int inputDepthDim, string activationFunction, int nrOfFilters, 
      int filterDim, int filterStride)
    {
      convolutionalLayer = new ConvolutionalLayer(inputDim, inputDepthDim, activationFunction, nrOfFilters, filterDim, filterStride);
      
      int featureMapDim = (inputDim - filterDim) / filterStride + 1;
      this.neuronsConvLayer = MiscLibrary.Initialize.AllocateDouble(featureMapDim, featureMapDim, nrOfFilters);

      this.weights[CONV_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfFilters, filterDim, filterDim, inputDepthDim);
      this.weightChanges[CONV_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfFilters, filterDim, filterDim, inputDepthDim);
      this.weightGradients[CONV_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfFilters, filterDim, filterDim, inputDepthDim);
      MiscLibrary.Initialize.SetToZero(this.weightChanges[CONV_LAYER_IND]);

      this.biases[CONV_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfFilters);
      this.biasChanges[CONV_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfFilters);
      this.biasGradients[CONV_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfFilters);
      MiscLibrary.Initialize.SetToZero(this.biasChanges[CONV_LAYER_IND]);
    }

    /// <summary>
    /// Constructs and initializes the weights biases and paramters for the convolution layer
    /// </summary>
    /// <param name="inputDim"></param>
    /// <param name="inputDepthDim"></param>
    /// <param name="activationFunction"></param>
    /// <param name="outputDim"></param>
    /// <param name="lossFunction"></param>
    public void InitializeFullyConnectedOutputLayer(int inputDim, int inputDepthDim, string activationFunction, 
      int outputDim, string lossFunction)
    {
      outputLayer = new FullyConnectedOutputLayer(inputDim, inputDepthDim, activationFunction, outputDim, lossFunction);

      int nrOfOutputs = outputDim * outputDim;
      this.neuronsOutputLayer = MiscLibrary.Initialize.AllocateDouble(nrOfOutputs);

      this.weights[OUTPUT_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfOutputs, inputDim, inputDim, inputDepthDim);
      this.weightChanges[OUTPUT_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfOutputs, inputDim, inputDim, inputDepthDim);
      this.weightGradients[OUTPUT_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfOutputs, inputDim, inputDim, inputDepthDim);
      MiscLibrary.Initialize.SetToZero(this.weightChanges[OUTPUT_LAYER_IND]);

      this.biases[OUTPUT_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfOutputs);
      this.biasChanges[OUTPUT_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfOutputs);
      this.biasGradients[OUTPUT_LAYER_IND] = MiscLibrary.Initialize.AllocateDouble(nrOfOutputs);
      MiscLibrary.Initialize.SetToZero(this.biasChanges[OUTPUT_LAYER_IND]);
    }
    #endregion


    // Training the network
    #region
    /// <summary>
    /// Trains the network with image pairs located named SatX.bmp and GoldX.bmp where X = {1,2,...,N}
    /// </summary>
    /// <param name="networkFolderLocation"></param>
    /// <param name="maxNrOfEpochs"></param>
    /// <param name="maxNrOfMinutes"></param>
    /// <param name="miniBatchSize"></param>
    /// <param name="initLearningRate"></param>
    /// <param name="finalLearningRate"></param>
    /// <param name="initMomentum"></param>
    /// <param name="finalMomentum"></param>
    /// <param name="weightDecay"></param>
    /// <param name="calculateErrorInterval"></param>
    /// <param name="updateMomentumLearingRateInterval"></param>
    /// <param name="nrOfTrainingImagePairs"></param>
    /// <param name="nrOfValidationImagePairs"></param>
    /// <param name="roadProbabilityThreshold"></param>
    /// <param name="visuallyCheckTraining"></param>
    public void TrainNetwork(string networkFolderLocation, int maxNrOfEpochs, int maxNrOfMinutes, int miniBatchSize, 
       double initLearningRate, double finalLearningRate, double initMomentum, double finalMomentum, 
      double weightDecay, int calculateErrorInterval, int updateMomentumLearingRateInterval, int nrOfTrainingImagePairs,
      int maxNrOfTrainingImagePairsInMemory, int nrOfValidationImagePairs, double roadProbabilityThreshold, bool visuallyCheckTraining)
    {
      // Fix directories
      #region
      string trainingSetFolder = "Training set\\", validationSetFolder = "Validation set\\", checkTrainingFolder = "Check training\\", 
        networkParameterFolder = "Network files\\", satImgStr = "Sat", imgExt = ".bmp";

      string trainingSetLocation = networkFolderLocation + trainingSetFolder;
      string validationSetLocation = networkFolderLocation + validationSetFolder;
      string checkTrainingLocation = networkFolderLocation + checkTrainingFolder;
      string networkParameterFolderPath = networkFolderLocation + networkParameterFolder;
      string logFileLocation = networkParameterFolderPath + "log.txt";

      Directory.CreateDirectory(checkTrainingLocation);
      Directory.CreateDirectory(networkParameterFolderPath);
      #endregion

      // Declare and initialize variables
      #region
      List<double[][][]> satTrainSetPatches = new List<double[][][]>();
      List<double[][][]> satValidationSetPatches = new List<double[][][]>();
      List<double[][][]> miniBatchInput;
      List<byte[]> goldTrainSetPatches = new List<byte[]>();
      List<byte[]> goldValidationSetPatches = new List<byte[]>();
      List<byte[]> miniBatchGold;
      List<int> randTrainImageInd, trainImageInd;
      List<int> validationImageInd;

      Bitmap checkSizeBitmap = new Bitmap(trainingSetLocation + satImgStr + "1" + imgExt);
      int inputImageHeight = checkSizeBitmap.Height;
      int inputImageWidth = checkSizeBitmap.Width;
      int outputImageHeight = inputImageHeight - 2 * this.outputOffset;
      int outputImageWidth = inputImageWidth - 2 * this.outputOffset;
      int nrOfHeightPatches = outputImageHeight / this.networkOutputDim;
      int nrOfWidthPatches = outputImageWidth / this.networkOutputDim; 
      int nrOfPatchesPerImage = nrOfHeightPatches * nrOfWidthPatches;

      int nrOfTrainingPatches = nrOfTrainingImagePairs * nrOfPatchesPerImage;
      int maxNrOfTrainingPatchesInMemory = maxNrOfTrainingImagePairsInMemory * nrOfPatchesPerImage;
      int nrOfTrainingImagesInMemory, nrOfTrainingPatchesInMemory;
      int nrOfWeightUpdatesPerEpoch = nrOfTrainingPatches / miniBatchSize;

      double[] currentError = new double[2]; ; // {Classification error, Loss}
      double[] bestClassificationError = {Double.MaxValue, Double.MaxValue};
      double[] bestLoss = {Double.MaxValue, Double.MaxValue};

      double learningRate = initLearningRate;
      double momentum = initMomentum;
      #endregion

      if (visuallyCheckTraining)
        Misc.SaveCheckTrainingImages(validationSetLocation, checkTrainingLocation, nrOfValidationImagePairs,
          inputImageWidth, inputImageHeight, outputImageWidth, outputImageHeight);
      
      stdDevTrainSet = Misc.GetStandardDeviation(trainingSetLocation, nrOfTrainingImagePairs, inputImageHeight, 
        inputImageWidth);
      
      // Import the full validation set in memory
      validationImageInd = Enumerable.Range(0, nrOfValidationImagePairs).ToList();
      Misc.LoadSetPatches(satValidationSetPatches, goldValidationSetPatches, validationSetLocation, validationImageInd, 
        nrOfHeightPatches, nrOfWidthPatches, networkInputDim, networkOutputDim, stdDevTrainSet, outputOffset);

      // Save inital information to log file
      #region
      using (System.IO.StreamWriter logFile = new System.IO.StreamWriter(logFileLocation, false)) {
        logFile.WriteLine("Network information:");
        logFile.WriteLine("Network input patch dim: {0}", networkInputDim);
        logFile.WriteLine("Network output patch dim: {0}", networkOutputDim);
        logFile.WriteLine("Number of filters: {0}", weightDim[CONV_LAYER_IND][0]);
        logFile.WriteLine("Filter dim: {0}", weightDim[CONV_LAYER_IND][1]);
        logFile.WriteLine("Convolutional layer activation function: " + convolutionalLayer.GetActivationFunction());
        logFile.WriteLine();
        logFile.WriteLine("Training information:");
        logFile.WriteLine("Inital learning rate: {0}", initLearningRate);
        logFile.WriteLine("Final learning rate: {0}", finalLearningRate);
        logFile.WriteLine("Inital momentum: {0}", initMomentum);
        logFile.WriteLine("Final momentum: {0}", finalMomentum);
        logFile.WriteLine("Weight Decay: {0}", weightDecay);
        logFile.WriteLine("Mini batch size: {0}", miniBatchSize);
        logFile.WriteLine("Update momentum and learning rate interval: {0}", updateMomentumLearingRateInterval);
        logFile.WriteLine("Max number of epochs: {0}", maxNrOfEpochs);
        logFile.WriteLine("Max number of minutes: {0}", maxNrOfMinutes);
        logFile.WriteLine();
      }
      #endregion

      // Three ending conditions: Max epoch, max time and press escape
      Random rng = new Random();
      DateTime startTime = DateTime.UtcNow;
      int l = 0;
      while((l < maxNrOfEpochs) && 
        (DateTime.UtcNow - startTime < TimeSpan.FromMinutes(maxNrOfMinutes)) &&
        !(Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Escape))
     {
        if (l % calculateErrorInterval == 0 || l == maxNrOfEpochs - 1) {
          currentError = CalculateTotalClassificationErrorAndLoss(satValidationSetPatches, goldValidationSetPatches,
            roadProbabilityThreshold);
          // Print to console and log file
          #region
          System.Console.WriteLine("Epoch: {0}", l);
          System.Console.WriteLine("Elapsed time: {0}", DateTime.UtcNow - startTime);
          System.Console.WriteLine("Total classification error: {0}", currentError[0]);
          System.Console.WriteLine("Total loss: {0}", currentError[1]);

          // Save to log file
          using (System.IO.StreamWriter logFile = new System.IO.StreamWriter(logFileLocation, true)) {
            logFile.WriteLine("Epoch: {0}", l);
            logFile.WriteLine("Elapsed time: {0}", DateTime.UtcNow - startTime);
            logFile.WriteLine("Total classification error: {0}", currentError[0]);
            logFile.WriteLine("Total loss: {0}", currentError[1]);
          }
          #endregion

          // Save best classification and loss network
          #region
          if (currentError[0] < bestClassificationError[0]) {
            bestClassificationError = currentError;
            string networkFilePath = networkParameterFolderPath + "BestClassificationErrorNetworkFile";
            SaveNetworkParameters(networkFilePath);

            System.Console.WriteLine("Best classification error so far achieved! Network saved.");
            using (System.IO.StreamWriter logFile = new System.IO.StreamWriter(logFileLocation, true)) {
              logFile.WriteLine("Best classification error so far achieved! Network saved.");
            }
          }
          if (currentError[1] <= bestLoss[1]) {
            bestLoss = currentError;
            string networkFilePath = networkParameterFolderPath + "BestLossNetworkFile";
            SaveNetworkParameters(networkFilePath);
            
            System.Console.WriteLine("Best loss so far achieved! Network saved.");
            using (System.IO.StreamWriter logFile = new System.IO.StreamWriter(logFileLocation, true)) {
              logFile.WriteLine("Best loss so far achieved! Network saved.");
            }
          }
          #endregion
        }

        // Code for saving current predictions of the validation set
        #region
        bool correctEpoch = l % calculateErrorInterval == 0 || l == 1 || l == maxNrOfEpochs - 1;
        if (visuallyCheckTraining && correctEpoch) {
          for (int i = 0; i < nrOfValidationImagePairs; ++i) {
            string rawRoadMapName = "RawRoadMap", imgFolderPath = "img" + (i + 1).ToString() + "\\";
            string rawRoadMapLocation = checkTrainingLocation + imgFolderPath + rawRoadMapName + (i + 1).ToString() + "_Epoch" + l.ToString() + imgExt;
            string satImgLocation = validationSetLocation + satImgStr + (i + 1).ToString() + imgExt;

            Bitmap rawRoadMap = GetRawRoadBitmap(satImgLocation);
            rawRoadMap.Save(rawRoadMapLocation);
          }
        }
        #endregion

        // Randomly select order for training images:
        randTrainImageInd = Enumerable.Range(0, nrOfTrainingImagePairs).OrderBy(r => rng.Next()).ToList();

        for (int i = 0; i < nrOfTrainingImagePairs; i += maxNrOfTrainingImagePairsInMemory) {
          // Import part of the training set in memory and shuffel it
          nrOfTrainingImagesInMemory = maxNrOfTrainingImagePairsInMemory < nrOfTrainingImagePairs - i ? maxNrOfTrainingImagePairsInMemory : nrOfTrainingImagePairs - i;
          nrOfTrainingPatchesInMemory = nrOfTrainingImagesInMemory * nrOfPatchesPerImage;
          trainImageInd = randTrainImageInd.GetRange(i, nrOfTrainingImagesInMemory);
          Misc.LoadSetPatches(satTrainSetPatches, goldTrainSetPatches, trainingSetLocation, trainImageInd, 
            nrOfHeightPatches, nrOfWidthPatches, networkInputDim, networkOutputDim, stdDevTrainSet, outputOffset);
          Misc.ShuffleTrainSet(satTrainSetPatches, goldTrainSetPatches, rng);

          for (int k = 0; k < nrOfTrainingPatchesInMemory; k += miniBatchSize) {
            // Retrieves a minibatch from training set
            int nrOfPatchesInMiniBatch = k + miniBatchSize < nrOfTrainingPatchesInMemory ? miniBatchSize : nrOfTrainingPatchesInMemory - k;
            miniBatchInput = satTrainSetPatches.GetRange(k, nrOfPatchesInMiniBatch);
            miniBatchGold = goldTrainSetPatches.GetRange(k, nrOfPatchesInMiniBatch);
            
            MiniBatchWeightUpdate(miniBatchInput, miniBatchGold, learningRate, momentum, weightDecay);
          }

          satTrainSetPatches.Clear();
          goldTrainSetPatches.Clear();
        }

        if (l % updateMomentumLearingRateInterval == 0)
          UpdateLearningRateAndMomentum(ref learningRate, finalLearningRate, ref momentum, finalMomentum);
        
        ++l;
      }


      // Code for saving current predictions of the validation set
      #region
      if (visuallyCheckTraining) {
        for (int i = 0; i < nrOfValidationImagePairs; ++i) {
          string rawRoadMapName = "RawRoadMap", imgFolderPath = "img" + (i + 1).ToString() + "\\";
          string rawRoadMapLocation = checkTrainingLocation + imgFolderPath + rawRoadMapName + (i + 1).ToString() + "_Epoch" + l.ToString() + imgExt;
          string satImgLocation = validationSetLocation + satImgStr + (i + 1).ToString() + imgExt;

          Bitmap rawRoadMap = GetRawRoadBitmap(satImgLocation);
          rawRoadMap.Save(rawRoadMapLocation);
        }
      }
      #endregion

      currentError = CalculateTotalClassificationErrorAndLoss(satValidationSetPatches, goldValidationSetPatches,
          roadProbabilityThreshold);
      // Print to console and log file
      #region
      System.Console.WriteLine("Epoch: {0}", l);
      System.Console.WriteLine("Elapsed time: {0}", DateTime.UtcNow - startTime);
      System.Console.WriteLine("Total classification error: {0}", currentError[0]);
      System.Console.WriteLine("Total loss: {0}", currentError[1]);

      // Save to log file
      using (System.IO.StreamWriter logFile = new System.IO.StreamWriter(logFileLocation, true)) {
        logFile.WriteLine("Epoch: {0}", l);
        logFile.WriteLine("Elapsed time: {0}", DateTime.UtcNow - startTime);
        logFile.WriteLine("Total classification error: {0}", currentError[0]);
        logFile.WriteLine("Total loss: {0}", currentError[1]);
      }
      #endregion

      // Save best classification and loss network
      #region
      if (currentError[0] < bestClassificationError[0]) {
        bestClassificationError = currentError;
        string networkFilePath = networkParameterFolderPath + "BestClassificationErrorNetworkFile";
        SaveNetworkParameters(networkFilePath);

        System.Console.WriteLine("Best classification error so far achieved! Network saved.");
        using (System.IO.StreamWriter logFile = new System.IO.StreamWriter(logFileLocation, true)) {
          logFile.WriteLine("Best classification error so far achieved! Network saved.");
        }
      }
      if (currentError[1] <= bestLoss[1]) {
        bestLoss = currentError;
        string networkFilePath = networkParameterFolderPath + "BestLossNetworkFile";
        SaveNetworkParameters(networkFilePath);

        System.Console.WriteLine("Best loss so far achieved! Network saved.");
        using (System.IO.StreamWriter logFile = new System.IO.StreamWriter(logFileLocation, true)) {
          logFile.WriteLine("Best loss so far achieved! Network saved.");
        }
      }
      #endregion

    }


    // Minibatch weight update functions
    #region
    /// <summary>
    /// Using stochastic gradient descent with minibatches, momentum and weight decay
    /// </summary>
    /// <param name="miniBatchInput"></param>
    /// <param name="miniBatchGold"></param>
    /// <param name="learningRate"></param>
    /// <param name="momentum"></param>
    /// <param name="weightDecay"></param>
    private void MiniBatchWeightUpdate(List<double[][][]> miniBatchInput, List<byte[]> miniBatchGold, double learningRate, 
      double momentum, double weightDecay)
    {
      int miniBatchSize = miniBatchInput.Count();

      double[] initialDeltas;
      double[][][] propagatedDeltas;

      MiscLibrary.Initialize.SetToZero(weightGradients[CONV_LAYER_IND]);
      MiscLibrary.Initialize.SetToZero(weightGradients[OUTPUT_LAYER_IND]);
      MiscLibrary.Initialize.SetToZero(biasGradients[CONV_LAYER_IND]);
      MiscLibrary.Initialize.SetToZero(biasGradients[OUTPUT_LAYER_IND]);

      for (int i = 0; i < miniBatchSize; ++i) {
        // Forward Propagate
        this.convolutionalLayer.ForwardPropagate(miniBatchInput[i], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
        this.outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);

        // Backward Propagate deltas
        initialDeltas = this.outputLayer.CalculateInitialDeltas(miniBatchGold[i], this.neuronsOutputLayer);
        propagatedDeltas = this.outputLayer.BackwardPropagate(this.weights[OUTPUT_LAYER_IND], this.neuronsConvLayer, this.convolutionalLayer.GetActivationFunction());

        // Calculates the weight and bias gradients and adds them to the running gradient sums
        this.outputLayer.AddWeightAndBiasGradients(initialDeltas, this.neuronsConvLayer, this.weightGradients[OUTPUT_LAYER_IND], this.biasGradients[OUTPUT_LAYER_IND]);
        this.convolutionalLayer.AddWeightAndBiasGradients(propagatedDeltas, miniBatchInput[i], this.weightGradients[CONV_LAYER_IND], this.biasGradients[CONV_LAYER_IND]);
      }
      this.UpdatingWeightsAndBiasesUsingSGD(learningRate, momentum, weightDecay);
    }


    /// <summary>
    /// Updating all of the weights and biases using stochastic gradient descent with momentum and L2 weight decay.
    /// Update formula: w(t+1) = w(t) + (momentum*w(t) - learningRate * (weightDecay * w(t) + dL/dw(t))
    /// </summary>
    /// <param name="learningRate"></param>
    /// <param name="momentum"></param>
    /// <param name="weightDecay"></param>
    private void UpdatingWeightsAndBiasesUsingSGD(double learningRate, double momentum, double weightDecay)
    {
      for (int indLayer = 0; indLayer < NR_OF_LAYERS; ++indLayer) {
        for (int k = 0; k < this.weightDim[indLayer][0]; ++k) {

          this.biasChanges[indLayer][k] = momentum * this.biasChanges[indLayer][k] - 
            learningRate * (biasGradients[indLayer][k] + weightDecay * this.biases[indLayer][k]);
          
          this.biases[indLayer][k] += this.biasChanges[indLayer][k];

          for (int i = 0; i < this.weightDim[indLayer][1]; ++i) {
            for (int j = 0; j < this.weightDim[indLayer][2]; ++j) {
              for (int f = 0; f < this.weightDim[indLayer][3]; ++f) {

                this.weightChanges[indLayer][k][i][j][f] = momentum * this.weightChanges[indLayer][k][i][j][f] - 
                  learningRate * (this.weightGradients[indLayer][k][i][j][f] + weightDecay * this.weights[indLayer][k][i][j][f]);
                
                this.weights[indLayer][k][i][j][f] += this.weightChanges[indLayer][k][i][j][f];
              }
            }
          }

        }
      }
    }


    /// <summary>
    /// Updating the momentum and learning rate
    /// </summary>
    /// <param name="learningRate"></param>
    /// <param name="finalLearningRate"></param>
    /// <param name="momentum"></param>
    /// <param name="finalMomentum"></param>
    private void UpdateLearningRateAndMomentum(ref double learningRate, double finalLearningRate, ref double momentum, double finalMomentum)
    {
      if (momentum < finalMomentum) {
        momentum += 0.1;
        momentum = momentum > finalMomentum ? finalMomentum : momentum;
      }

      if (learningRate < finalLearningRate) {
        learningRate /= 2;
        learningRate = learningRate > finalLearningRate ? finalLearningRate : learningRate;
      }
    }
    #endregion


    // Calculate classification error and loss methods
    #region

    /// <summary>
    /// Calculates the total classification error and loss for the whole training set
    /// </summary>
    /// <param name="satTrainSetPatches"></param>
    /// <param name="goldTrainSetPatches"></param>
    /// <param name="roadProbabilityThreshold"></param>
    /// <returns></returns>
    private double[] CalculateTotalClassificationErrorAndLoss(List<double[][][]> satTrainSetPatches, 
      List<byte[]> goldTrainSetPatches, double roadProbabilityThreshold)
    {
      double[] error = new double[2];
      int totalClassificationError = 0;
      double totalLoss = 0;
      int nrOfPatches = satTrainSetPatches.Count();

      for (int i = 0; i < nrOfPatches; ++i) {
        // Forward Propagate
        this.convolutionalLayer.ForwardPropagate(satTrainSetPatches[i], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
        this.outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);

        totalClassificationError += outputLayer.GetCurrentClassificationError(goldTrainSetPatches[i], this.neuronsOutputLayer, roadProbabilityThreshold);
        totalLoss += outputLayer.GetCurrentLoss(goldTrainSetPatches[i], this.neuronsOutputLayer);
      }

      error[0] = totalClassificationError;
      error[1] = totalLoss;

      return error;
    }


    /// <summary>
    /// Returns the classification error (number of pixels predicted wrong) for a single image
    /// </summary>
    /// <param name="predictedRoadMapLocation"></param>
    /// <param name="goldImageLocation"></param>
    /// <returns></returns>
    public int CalculateSingleImageClassificationError(string predictedRoadMapLocation, string goldImageLocation)
    {
      int singleImageClassificationError = 0;

      byte[][][] goldPixelValues;
      byte[][][] predictedRoadMapValues;
      byte binary_threshold = 20;

      predictedRoadMapValues = MiscLibrary.ImageProcessor.GetPixelValues(predictedRoadMapLocation);
      goldPixelValues = MiscLibrary.ImageProcessor.GetPixelValues(goldImageLocation);
      MiscLibrary.ImageProcessor.ConvertToStandardGrayScale(goldPixelValues);
      MiscLibrary.ImageProcessor.GetRoadPixelClassificationData(binary_threshold, goldPixelValues);

      for (int i = 0; i < predictedRoadMapValues.Length; ++i) {
        for (int j = 0; j < predictedRoadMapValues[0].Length; ++j) {
          singleImageClassificationError += Math.Abs(predictedRoadMapValues[i][j][0] - goldPixelValues[i + outputOffset][j + outputOffset][0]);
        }
      }

      return singleImageClassificationError;
    }
    #endregion
    #endregion


    // Return predicted road map functions
    #region

    /// <summary>
    /// Returns the road map of a satellite image
    /// </summary>
    /// <param name="satImgLocation"></param>
    /// <returns></returns>
    public Bitmap GetRoadBitmap(string inputImgLocation, double roadProbabilityThreshold)
    {
      Bitmap predictedBitmap;

      byte[][][] satPixelValues;
      byte[][][] predictedRoadMapValues;

      List<double[][][]> satImgPatches;
      List<byte[]> predictedRoadMapPatches = new List<byte[]>();
      Bitmap inputImg = new Bitmap(inputImgLocation);
      int outputImageHeight = inputImg.Height - 2 * this.outputOffset;
      int outputImageWidth = inputImg.Width - 2 * this.outputOffset;
      int nrOfHeightPatches = outputImageHeight / this.networkOutputDim;
      int nrOfWidthPatches = outputImageWidth / this.networkOutputDim;

      satPixelValues = MiscLibrary.ImageProcessor.GetPixelValues(inputImg);
      satImgPatches = Misc.GetSingleSatImgPatches(satPixelValues, nrOfHeightPatches, nrOfWidthPatches, 
        this.networkInputDim, this.networkOutputDim, this.stdDevTrainSet);
      
      int nrOfImagePatches = satImgPatches.Count();
      for (int i = 0; i < nrOfImagePatches; ++i) {
        // Forward Propagate
        this.convolutionalLayer.ForwardPropagate(satImgPatches[i], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
        this.outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);

        predictedRoadMapPatches.Add(this.outputLayer.GetCurrentPredictedRoadPatch(this.neuronsOutputLayer, roadProbabilityThreshold));
      }

      predictedRoadMapValues = ReshapeOutputPatchesToPixelValues(predictedRoadMapPatches, outputImageHeight, outputImageWidth);
      predictedBitmap = MiscLibrary.ImageProcessor.GetBitmap(predictedRoadMapValues);

      return predictedBitmap;
    }



    /// <summary>
    /// Returns the road map of a satellite image
    /// </summary>
    /// <param name="satImgLocation"></param>
    /// <returns></returns>
    public Bitmap GetRoadBitmap(Bitmap inputImg, double roadProbabilityThreshold)
    {
      Bitmap predictedBitmap;

      byte[][][] satPixelValues;
      byte[][][] predictedRoadMapValues;

      List<double[][][]> satImgPatches;
      List<byte[]> predictedRoadMapPatches = new List<byte[]>();
      int outputImageHeight = inputImg.Height - 2 * this.outputOffset;
      int outputImageWidth = inputImg.Width - 2 * this.outputOffset;
      int nrOfHeightPatches = outputImageHeight / this.networkOutputDim;
      int nrOfWidthPatches = outputImageWidth / this.networkOutputDim;

      satPixelValues = MiscLibrary.ImageProcessor.GetPixelValues(inputImg);
      satImgPatches = Misc.GetSingleSatImgPatches(satPixelValues, nrOfHeightPatches, nrOfWidthPatches,
        this.networkInputDim, this.networkOutputDim, this.stdDevTrainSet);

      int nrOfImagePatches = satImgPatches.Count();
      for (int i = 0; i < nrOfImagePatches; ++i) {
        // Forward Propagate
        this.convolutionalLayer.ForwardPropagate(satImgPatches[i], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
        this.outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);

        predictedRoadMapPatches.Add(this.outputLayer.GetCurrentPredictedRoadPatch(this.neuronsOutputLayer, roadProbabilityThreshold));
      }

      predictedRoadMapValues = ReshapeOutputPatchesToPixelValues(predictedRoadMapPatches, outputImageHeight, outputImageWidth);
      predictedBitmap = MiscLibrary.ImageProcessor.GetBitmap(predictedRoadMapValues);

      return predictedBitmap;
    }



    /// <summary>
    /// Return the road map of the satImage
    /// </summary>
    /// <param name="satImgLocation"></param>
    /// <param name="roadMapLocation"></param>
    /// <returns></returns>
    public Bitmap GetRawRoadBitmap(string inputImgLocation)
    {
      Bitmap predictedRawBitmap;

      byte[][][] satPixelValues;
      byte[][][] predictedRawRoadMapValues;

      List<double[][][]> satImgPatches;
      List<byte[]> predictedRoadMapPatches = new List<byte[]>();
      Bitmap inputImg = new Bitmap(inputImgLocation);
      int outputImageHeight = inputImg.Height - 2 * this.outputOffset;
      int outputImageWidth = inputImg.Width - 2 * this.outputOffset;
      int nrOfHeightPatches = outputImageHeight / this.networkOutputDim;
      int nrOfWidthPatches = outputImageWidth / this.networkOutputDim;

      satPixelValues = MiscLibrary.ImageProcessor.GetPixelValues(inputImg);
      satImgPatches = Misc.GetSingleSatImgPatches(satPixelValues, nrOfHeightPatches, nrOfWidthPatches,
        networkInputDim, networkOutputDim, stdDevTrainSet);

      int nrOfImagePatches = satImgPatches.Count();
      for (int i = 0; i < nrOfImagePatches; ++i) {
        // Forward Propagate
        convolutionalLayer.ForwardPropagate(satImgPatches[i], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
        outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);

        predictedRoadMapPatches.Add(this.outputLayer.GetCurrentPredictedRawRoadPatch(this.neuronsOutputLayer));
      }

      predictedRawRoadMapValues = ReshapeOutputPatchesToPixelValues(predictedRoadMapPatches, outputImageHeight, outputImageWidth);
      predictedRawBitmap = MiscLibrary.ImageProcessor.GetBitmap(predictedRawRoadMapValues);

      return predictedRawBitmap;
    }



    /// <summary>
    /// Returns the road map pixel values from the pixel values of a satellite image  
    /// </summary>
    /// <param name="pixelValues"></param>
    /// <returns></returns>
    public byte[][][] GetRoadMapPixelValues(byte[][][] satPixelValues, double roadProbabilityThreshold)
    {
      byte[][][] predictedRoadMapValues;

      List<double[][][]> satImgPatches;
      List<byte[]> predictedRoadMapPatches = new List<byte[]>();

      int inputImageHeight = satPixelValues.Length;
      int inputImageWidth = satPixelValues[0].Length;
      int outputImageHeight = inputImageHeight - 2 * this.outputOffset;
      int outputImageWidth = inputImageWidth - 2 * this.outputOffset;
      int nrOfHeightPatches = outputImageHeight / this.networkOutputDim;
      int nrOfWidthPatches = outputImageWidth / this.networkOutputDim; 

      satImgPatches = Misc.GetSingleSatImgPatches(satPixelValues, nrOfHeightPatches, nrOfWidthPatches,
        this.networkInputDim, this.networkOutputDim, this.stdDevTrainSet);

      int nrOfImagePatches = satImgPatches.Count();
      for (int i = 0; i < nrOfImagePatches; ++i) {
        // Forward Propagate
        this.convolutionalLayer.ForwardPropagate(satImgPatches[i], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
        this.outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);

        predictedRoadMapPatches.Add(this.outputLayer.GetCurrentPredictedRoadPatch(this.neuronsOutputLayer, roadProbabilityThreshold));
      }

      predictedRoadMapValues = ReshapeOutputPatchesToPixelValues(predictedRoadMapPatches, outputImageHeight, outputImageWidth);

      return predictedRoadMapValues;
    }



    /// <summary>
    /// Reshapes the output patches to a pixel value image
    /// </summary>
    /// <param name="outputMiniBatch"></param>
    /// <returns></returns>
    private byte[][][] ReshapeOutputPatchesToPixelValues(List<byte[]> outputMiniBatch, int outputImageHeight, int outputImageWidth)
    {
      byte[][][] pixelValues = MiscLibrary.Initialize.AllocateByte(outputImageHeight, outputImageWidth, NR_OF_PIXEL_CHANNELS);

      int redPixelChannel = 0, greenPixelChannel = 1, bluePixelChannel = 2;
      int batchInd, rowInd, colInd, batchPixelInd;
      int startRowInd, startColInd;
      int nrOfHeightPatches = outputImageHeight / this.networkOutputDim;
      int nrOfWidthPatches = outputImageWidth / this.networkOutputDim;

      for (int i = 0; i < nrOfHeightPatches; ++i) {
        for (int j = 0; j < nrOfWidthPatches; ++j) {
          batchInd = i * nrOfWidthPatches + j;
          startRowInd = i * this.networkOutputDim;
          startColInd = j * this.networkOutputDim;

          for (int ii = 0; ii < this.networkOutputDim; ++ii) {
            for (int jj = 0; jj < this.networkOutputDim; ++jj) {
              rowInd = startRowInd + ii;
              colInd = startColInd + jj;
              batchPixelInd = ii * networkOutputDim + jj;

              pixelValues[rowInd][colInd][redPixelChannel] = outputMiniBatch[batchInd][batchPixelInd];
              pixelValues[rowInd][colInd][greenPixelChannel] = outputMiniBatch[batchInd][batchPixelInd];
              pixelValues[rowInd][colInd][bluePixelChannel] = outputMiniBatch[batchInd][batchPixelInd];
            }
          }

        }
      }
      return pixelValues;
    }


    #endregion


    // Saving methods
    #region
    /// <summary>
    /// Saving the network parameters: standard deviation, weights and biases to a file
    /// </summary>
    /// <param name="networkFilePath"></param>
    public void SaveNetworkParameters(string networkFilePath)
    {
      BinaryFormatter bf = new BinaryFormatter();
      using (FileStream stream = new FileStream(networkFilePath, FileMode.Create, FileAccess.Write)) {
        bf.Serialize(stream, stdDevTrainSet);
        bf.Serialize(stream, weights);
        bf.Serialize(stream, biases);
      }
    }
    #endregion


    // Loading methods
    #region 
    /// <summary>
    /// Initializes all of the weight and biases of the network to random values.
    /// </summary>
    public void InitializeRandomWeightAndBiases()
    {
      convolutionalLayer.InitializeWeightsAndBiases(weights[CONV_LAYER_IND], biases[CONV_LAYER_IND]);
      outputLayer.InitializeWeightsAndBiases(weights[OUTPUT_LAYER_IND], biases[OUTPUT_LAYER_IND]);
    }


    /// <summary>
    /// Initializes a network using a network file
    /// </summary>
    /// <param name="networkFilePath"></param>
    public void LoadNetworkParameters(string networkFilePath)
    {
      BinaryFormatter bf = new BinaryFormatter();
      using (FileStream stream = new FileStream(networkFilePath, FileMode.Open, FileAccess.Read)) {
        stdDevTrainSet = (double)bf.Deserialize(stream);
        weights = (double[][][][][])bf.Deserialize(stream);
        biases = (double[][])bf.Deserialize(stream);
      }
    }
    #endregion


    // Testing methods
    #region
    /// <summary>
    /// Testing the network backpropagation using the finite difference method for a few epsilon cases
    /// </summary>
    /// <param name="satImgLocation"></param>
    /// <param name="goldImgLocation"></param>
    public void TestNetwork(string satImgLocation, string goldImgLocation)
    {
      this.stdDevTrainSet = 1; // just for testing

      byte[][][] satPixelValues = MiscLibrary.ImageProcessor.GetPixelValues(satImgLocation);
      byte[][][] goldPixelValues = MiscLibrary.ImageProcessor.GetPixelValues(goldImgLocation);

      List<double[][][]> satImgPatches;
      List<byte[]> goldImgPatches;

      int inputImageHeight = satPixelValues.Length;
      int inputImageWidth = satPixelValues[0].Length;
      int outputImageHeight = inputImageHeight - 2 * this.outputOffset;
      int outputImageWidth = inputImageWidth - 2 * this.outputOffset;
      int nrOfHeightPatches = outputImageHeight / this.networkOutputDim;
      int nrOfWidthPatches = outputImageWidth / this.networkOutputDim; 

      satImgPatches = Misc.GetSingleSatImgPatches(satPixelValues, nrOfHeightPatches, nrOfWidthPatches, 
        this.networkInputDim, this.networkOutputDim, this.stdDevTrainSet);
      goldImgPatches = Misc.GetSingleGoldImgPatches(goldPixelValues, nrOfHeightPatches, nrOfWidthPatches, this.networkInputDim,
        this.networkOutputDim, this.outputOffset);
      
      double[] initialDeltas;
      double[][][] propagatedDeltas;

      MiscLibrary.Initialize.SetToZero(weightGradients[CONV_LAYER_IND]);
      MiscLibrary.Initialize.SetToZero(weightGradients[OUTPUT_LAYER_IND]);
      MiscLibrary.Initialize.SetToZero(biasGradients[CONV_LAYER_IND]);
      MiscLibrary.Initialize.SetToZero(biasGradients[OUTPUT_LAYER_IND]);

      int nrOfImagePatches = satImgPatches.Count();
      Random rng = new Random();
      int testImagePatchInd = rng.Next(0, nrOfImagePatches);

      // Using backpropagation one time:
      #region
      // Forward Propagate
      this.convolutionalLayer.ForwardPropagate(satImgPatches[testImagePatchInd], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
      this.outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);

      // Backward Propagate deltas
      initialDeltas = this.outputLayer.CalculateInitialDeltas(goldImgPatches[testImagePatchInd], this.neuronsOutputLayer);
      propagatedDeltas = this.outputLayer.BackwardPropagate(this.weights[OUTPUT_LAYER_IND], this.neuronsConvLayer, this.convolutionalLayer.GetActivationFunction());

      // Calculates the weight and bias gradients and adds them to the running gradient sums
      this.outputLayer.AddWeightAndBiasGradients(initialDeltas, this.neuronsConvLayer, this.weightGradients[OUTPUT_LAYER_IND], this.biasGradients[OUTPUT_LAYER_IND]);
      this.convolutionalLayer.AddWeightAndBiasGradients(propagatedDeltas, satImgPatches[testImagePatchInd], this.weightGradients[CONV_LAYER_IND], this.biasGradients[CONV_LAYER_IND]);
      #endregion

      double weightGradientFiniteDifference;
      double biasGradientFiniteDifference;
      double addedEpsilonLoss;
      double subtractedEpsilonLoss;
      double relativeGradientDifference;
      int k, i, j, f;

      for (int layerInd = 0; layerInd < NR_OF_LAYERS; ++layerInd) {
        // Disturbing a weight with epsilon
        k = rng.Next(0, weightDim[layerInd][0]);
        i = rng.Next(0, weightDim[layerInd][1]);
        j = rng.Next(0, weightDim[layerInd][2]);
        f = rng.Next(0, weightDim[layerInd][3]);

        // Using finite difference method (L(w + epsilon) - L(w - epsilon)) / (2epsilon)
        for (double epsilon = 0.1; epsilon > 0.0000001; epsilon /= 10) {
          // Checking weight backpropagation
          #region
          // Forward Propagate with added epsilon
          this.weights[layerInd][k][i][j][f] += epsilon;
          this.convolutionalLayer.ForwardPropagate(satImgPatches[testImagePatchInd], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
          this.outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);
          addedEpsilonLoss = this.outputLayer.GetCurrentLoss(goldImgPatches[testImagePatchInd], this.neuronsOutputLayer);

          // Forward Propagate with subtracted epsilon
          this.weights[layerInd][k][i][j][f] -= 2 * epsilon;
          this.convolutionalLayer.ForwardPropagate(satImgPatches[testImagePatchInd], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
          this.outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);
          subtractedEpsilonLoss = this.outputLayer.GetCurrentLoss(goldImgPatches[testImagePatchInd], this.neuronsOutputLayer);

          weightGradientFiniteDifference = (addedEpsilonLoss - subtractedEpsilonLoss) / (2 * epsilon);
          relativeGradientDifference = Math.Abs(this.weightGradients[layerInd][k][i][j][f] - weightGradientFiniteDifference) / (Math.Abs(this.weightGradients[layerInd][k][i][j][f]) + Math.Abs(weightGradientFiniteDifference));

          // Restore perturbed weight
          this.weights[layerInd][k][i][j][f] += epsilon;

          System.Console.WriteLine("Weight gradient difference for epsilon value of {0}: ", epsilon);
          System.Console.WriteLine("Weight Ind {0},{1},{2},{3},{4}: {5}", layerInd, k, i, j, f, this.weights[layerInd][k][i][j][f]);
          System.Console.WriteLine("Weight gradient backpropagation: {0}", this.weightGradients[layerInd][k][i][j][f]);
          System.Console.WriteLine("Weight gradient finite difference: {0}", weightGradientFiniteDifference);
          System.Console.WriteLine("Relative weight gradient difference: {0}", relativeGradientDifference);
          System.Console.WriteLine("");

          #endregion

          // Checking bias backpropagtion
          #region

          // Forward Propagate with added epsilon
          this.biases[layerInd][k] += epsilon;
          this.convolutionalLayer.ForwardPropagate(satImgPatches[testImagePatchInd], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
          this.outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);
          addedEpsilonLoss = this.outputLayer.GetCurrentLoss(goldImgPatches[testImagePatchInd], this.neuronsOutputLayer);

          // Forward Propagate with subtracted epsilon
          this.biases[layerInd][k] -= 2 * epsilon;
          this.convolutionalLayer.ForwardPropagate(satImgPatches[testImagePatchInd], this.neuronsConvLayer, this.weights[CONV_LAYER_IND], this.biases[CONV_LAYER_IND]);
          this.outputLayer.ForwardPropagate(this.neuronsConvLayer, this.neuronsOutputLayer, this.weights[OUTPUT_LAYER_IND], this.biases[OUTPUT_LAYER_IND]);
          subtractedEpsilonLoss = this.outputLayer.GetCurrentLoss(goldImgPatches[testImagePatchInd], this.neuronsOutputLayer);

          biasGradientFiniteDifference = (addedEpsilonLoss - subtractedEpsilonLoss) / (2 * epsilon);
          relativeGradientDifference = Math.Abs(this.biasGradients[layerInd][k] - biasGradientFiniteDifference) / (Math.Abs(this.biasGradients[layerInd][k]) + Math.Abs(biasGradientFiniteDifference));

          // Restore perturbed bias
          this.biases[layerInd][k] += epsilon;

          System.Console.WriteLine("Bias gradient difference for epsilon value of {0}: ", epsilon);
          System.Console.WriteLine("Bias Ind {0},{1}: {2}", layerInd, k, this.biases[layerInd][k]);
          System.Console.WriteLine("Bias gradient backpropagation: {0}", this.biasGradients[layerInd][k]);
          System.Console.WriteLine("Bias gradient finite difference: {0}", biasGradientFiniteDifference);
          System.Console.WriteLine("Relative bias gradient difference: {0}", relativeGradientDifference);
          System.Console.WriteLine("");
          #endregion
        }
      }

    }
    #endregion

  }
   
}
