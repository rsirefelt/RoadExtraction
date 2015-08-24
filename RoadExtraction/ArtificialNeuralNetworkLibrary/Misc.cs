using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;


namespace ArtificialNeuralNetworkLibrary
{
  public static class Misc
  {
    private const byte NR_OF_PIXEL_CHANNELS = 3;
   

    /// <summary>
    /// Reads the satellite and gold images located in "selLocation" named Sat[ImageInd].bmp Gold[ImageInd].bmp and 
    /// fills the List<double[][][]> and List<byte[]> with all of the satellite and gold patches.
    /// </summary>
    /// <param name="satPatches"></param>
    /// <param name="goldPatches"></param>
    /// <param name="setLocation"></param>
    /// <param name="nrOfImagePairs"></param>
    /// <param name="nrOfHeightPatches"></param>
    /// <param name="nrOfWidthPatches"></param>
    /// <param name="networkInputDim"></param>
    /// <param name="networkOutputDim"></param>
    /// <param name="standardDeviationTrainSet"></param>
    /// <param name="outputOffset"></param>
    public static void LoadSetPatches(List<double[][][]> satPatches, List<byte[]> goldPatches,
      string setLocation, List<int> imageInds, int nrOfHeightPatches, int nrOfWidthPatches, int networkInputDim, 
      int networkOutputDim, double standardDeviationTrainSet, int outputOffset)
    {
      string satImgStr = "Sat", goldImgStr = "Gold", imgExt = ".bmp";
      string satImgLocation, goldImgLocation;
      byte binaryThreshold = 20;
      
      List<double[][][]> singleSatImgPatches = new List<double[][][]>();
      List<byte[]> singleGoldImgPatches = new List<byte[]>();

      byte[][][] satPixelValues, goldPixelValues;
      Bitmap satBitmap, goldBitmap;

      foreach (int imageInd in imageInds) {
        satImgLocation = setLocation + satImgStr + (imageInd + 1).ToString() + imgExt;
        goldImgLocation = setLocation + goldImgStr + (imageInd + 1).ToString() + imgExt;
        satBitmap = new Bitmap(satImgLocation);
        goldBitmap = new Bitmap(goldImgLocation);

        satPixelValues = MiscLibrary.ImageProcessor.GetPixelValues(satBitmap);
        singleSatImgPatches = GetSingleSatImgPatches(satPixelValues, nrOfHeightPatches, nrOfWidthPatches, networkInputDim, 
          networkOutputDim, standardDeviationTrainSet);

        // Converts to standard grey scale and binarize to remove grey objects from input images
        goldPixelValues = MiscLibrary.ImageProcessor.GetPixelValues(goldBitmap);
        MiscLibrary.ImageProcessor.ConvertToStandardGrayScale(goldPixelValues);
        MiscLibrary.ImageProcessor.GetRoadPixelClassificationData(binaryThreshold, goldPixelValues);
        singleGoldImgPatches = GetSingleGoldImgPatches(goldPixelValues, nrOfHeightPatches, nrOfWidthPatches, 
          networkInputDim, networkOutputDim, outputOffset);

        satPatches.AddRange(singleSatImgPatches);
        goldPatches.AddRange(singleGoldImgPatches);
      }
    }


    /// <summary>
    /// Shuffels the elements of the training set lists
    /// </summary>
    /// <param name="satTrainSetPatches"></param>
    /// <param name="goldTrainSetPatches"></param>
    /// <param name="rng"></param>
    public static void ShuffleTrainSet(List<double[][][]> satTrainSetPatches, List<byte[]> goldTrainSetPatches , Random rng)
    {
      int n = satTrainSetPatches.Count;
      if (n != goldTrainSetPatches.Count) 
        throw new ArgumentOutOfRangeException("The satellite train set needs to have the same size as the gold train set.");

      while (n > 1) {
        n--;
        int k = rng.Next(n + 1);
        double[][][] satValue = satTrainSetPatches[k];
        byte[] goldValue = goldTrainSetPatches[k];

        satTrainSetPatches[k] = satTrainSetPatches[n];
        goldTrainSetPatches[k] = goldTrainSetPatches[n];

        satTrainSetPatches[n] = satValue;
        goldTrainSetPatches[n] = goldValue;
      }
    }


    /// <summary>
    /// Randomly rotates every patch pair either 90, -90 or 180 degrees
    /// </summary>
    /// <param name="satTrainSetPatches"></param>
    /// <param name="goldTrainSetPatches"></param>
    /// <param name="rng"></param>
    public static void RandomlySemiRotateTrainSet(List<double[][][]> satTrainSetPatches, List<byte[]> goldTrainSetPatches, Random rng)
    {
      Parallel.For(0, satTrainSetPatches.Count, k => {
        byte rotationType = (byte)rng.Next(4);

        if (Convert.ToBoolean(rotationType)) {
          MiscLibrary.Mathematics.Rotate3dTensor(satTrainSetPatches[k], rotationType);
          MiscLibrary.Mathematics.RotateArray(goldTrainSetPatches[k], rotationType);
        }

      });
    }


    /// <summary>
    /// Takes the all of the pixel values in a satellite image and returns a List<double[][][]> containing 
    /// all of the preprocessed satellite image input patches. The preprocessing consists of subtracting 
    /// the mean value of every pixel patch and divide by the standard deviation from the entire training set. 
    /// </summary>
    /// <param name="satPixelValues">Pixel values for a satellite image</param>
    /// <param name="nrOfHeightPatches">Number of patches in the y-direction</param>
    /// <param name="nrOfWidthPatches">Number of patches in the x-direction</param>
    /// <param name="networkInputDim">Size of the network input dimension, determines patch size</param>
    /// <param name="networkOutputDim">Size of the network output dimension</param>
    /// <param name="standardDeviationTrainSet">The standard deviation for the entire training set</param>
    /// <returns></returns>
    public static List<double[][][]> GetSingleSatImgPatches(byte[][][] satPixelValues, int nrOfHeightPatches, int nrOfWidthPatches, 
      int networkInputDim, int networkOutputDim, double standardDeviationTrainSet)
    {
      List<double[][][]> singleSatImgPatches = new List<double[][][]>();
      int inputStartIndRow, inputEndIndRow, inputStartIndCol;

      double meanValue = 0;

      int batchInd = 0;
      for (int i = 0; i < nrOfHeightPatches; ++i) {
        inputStartIndRow = i * networkOutputDim;
        inputEndIndRow = inputStartIndRow + networkInputDim - 1;
        for (int j = 0; j < nrOfWidthPatches; ++j) {
          inputStartIndCol = j * networkOutputDim;

          singleSatImgPatches.Add(MiscLibrary.ImageProcessor.GetInputImagePatch(satPixelValues, inputStartIndRow, inputEndIndRow, inputStartIndCol));

          meanValue = MiscLibrary.Mathematics.CalculateMean(singleSatImgPatches[batchInd]);
          MiscLibrary.Mathematics.SubtractByValue(singleSatImgPatches[batchInd], meanValue);
          MiscLibrary.Mathematics.DivideByValue(singleSatImgPatches[batchInd], standardDeviationTrainSet);

          ++batchInd;
        }
      }

      return singleSatImgPatches;
    }


    /// <summary>
    /// Takes the all of the pixel values in a gold image and returns a List<byte[]> containing 
    /// all of the gold image input patches. The patches have one dimension because of easier 
    /// indexing when using the neural network. I.e. a 256 array corresponds to a 16x16 output patch.
    /// </summary>
    /// <param name="goldPixelValues">Pixel values for a gold image</param>
    /// <param name="nrOfHeightPatches">Number of patches in the y-direction</param>
    /// <param name="nrOfWidthPatches">Number of patches in the x-direction</param>
    /// <param name="networkInputDim">Size of the network input dimension, determines patch size</param>
    /// <param name="networkOutputDim">Size of the network output dimension</param>
    /// <param name="outputOffset"></param>
    /// <returns></returns>
    public static List<byte[]> GetSingleGoldImgPatches(byte[][][] goldPixelValues, int nrOfHeightPatches, int nrOfWidthPatches, 
      int networkInputDim, int networkOutputDim, int outputOffset)
    {
      List<byte[]> singleGoldImgPatches = new List<byte[]>();
      int startRowInd, endRowInd, startColInd;

      for (int i = 0; i < nrOfHeightPatches; ++i) {
        startRowInd = i * networkOutputDim + outputOffset;
        endRowInd = startRowInd + networkOutputDim - 1;

        for (int j = 0; j < nrOfWidthPatches; ++j) {
          startColInd = j * networkOutputDim + outputOffset;

          singleGoldImgPatches.Add(MiscLibrary.ImageProcessor.GetOutputImagePatch(goldPixelValues, startRowInd, endRowInd, startColInd));
        }
      }

      return singleGoldImgPatches;
    }


    /// <summary>
    /// Calculates the standard deviation over the entire data set.
    /// </summary>
    /// <param name="imageSetLocation"></param>
    /// <param name="nrOfImagePairs"></param>
    /// <param name="imageHeight"></param>
    /// <param name="imageWidth"></param>
    /// <returns></returns>
    public static double GetStandardDeviation(string imageSetLocation, int nrOfImagePairs, int imageHeight, int imageWidth)
    {
      double standardDeviation, meanValue = 0, variance = 0;
      int totalNrOfPixels = nrOfImagePairs * imageHeight * imageWidth * NR_OF_PIXEL_CHANNELS;

      string satImgStr = "Sat", imgExt = ".bmp";
      string satImgLocation;

      byte[][][] pixelValuesSat;

      // Calculates the total mean value:
      for (int imageInd = 1; imageInd < nrOfImagePairs + 1; ++imageInd) {
        satImgLocation = imageSetLocation + satImgStr + imageInd.ToString() + imgExt;
        pixelValuesSat = MiscLibrary.ImageProcessor.GetPixelValues(satImgLocation);

        for (int i = 0; i < imageHeight; ++i) {
          for (int j = 0; j < imageWidth; ++j) {
            for (int k = 0; k < NR_OF_PIXEL_CHANNELS; ++k) {
              meanValue += pixelValuesSat[i][j][k];
            }
          }
        }

      }
      meanValue /= totalNrOfPixels;

      // Calculating the variance in the data
      for (int imageInd = 1; imageInd < nrOfImagePairs + 1; ++imageInd) {
        satImgLocation = imageSetLocation + satImgStr + imageInd.ToString() + imgExt;
        pixelValuesSat = MiscLibrary.ImageProcessor.GetPixelValues(satImgLocation);

        for (int i = 0; i < imageHeight; ++i) {
          for (int j = 0; j < imageWidth; ++j) {
            for (int k = 0; k < NR_OF_PIXEL_CHANNELS; ++k) {
              variance += pixelValuesSat[i][j][k] * pixelValuesSat[i][j][k];
            }
          }
        }

      }
      variance = (variance / totalNrOfPixels) - (meanValue * meanValue);
      standardDeviation = Math.Sqrt(variance);

      return standardDeviation;
    }


    /// <summary>
    /// Save correctly cropped images (SatX.bmp and Gold.bmp) to separate folders used for 
    /// comparion during the training progress
    /// </summary>
    /// <param name="inputSetLocation"></param>
    /// <param name="checkTrainingLocation"></param>
    /// <param name="nrOfImagePairs"></param>
    /// <param name="inputImageWidth"></param>
    /// <param name="inputImageHeight"></param>
    /// <param name="outputImageWidth"></param>
    /// <param name="outputImageHeight"></param>
    public static void SaveCheckTrainingImages(string inputSetLocation, string checkTrainingLocation, 
      int nrOfImagePairs, int inputImageWidth, int inputImageHeight, int outputImageWidth, int outputImageHeight)
    {
      for (int i = 0; i < nrOfImagePairs; ++i) {
        string satImgName = "Sat", goldImgName = "Gold", imgExt = ".bmp", imgFolderPath = "img" + (i + 1).ToString() + "\\";

        string checkTrainImgFolderLocation = checkTrainingLocation + imgFolderPath;
        Directory.CreateDirectory(checkTrainImgFolderLocation);

        string satInputLocation = inputSetLocation + satImgName + (i + 1).ToString() + imgExt;
        string goldInputLocation = inputSetLocation + goldImgName + (i + 1).ToString() + imgExt;
        string satCheckTrainImgLocation = checkTrainImgFolderLocation + satImgName + (i + 1).ToString() + imgExt;
        string goldCheckTrainImgLocation = checkTrainImgFolderLocation + goldImgName + (i + 1).ToString() + imgExt;

        Bitmap satValidationBitmap = new Bitmap(satInputLocation);
        Bitmap satCheckTrainingBitmap = new Bitmap(satInputLocation);
        Bitmap goldCheckTrainingBitmap = new Bitmap(goldInputLocation);

        satValidationBitmap = MiscLibrary.ImageProcessor.CropBitmap(satValidationBitmap, inputImageWidth, inputImageHeight);
        satCheckTrainingBitmap = MiscLibrary.ImageProcessor.CropBitmap(satCheckTrainingBitmap, outputImageWidth, outputImageHeight);
        goldCheckTrainingBitmap = MiscLibrary.ImageProcessor.CropBitmap(goldCheckTrainingBitmap, outputImageWidth, outputImageHeight);

        satCheckTrainingBitmap.Save(satCheckTrainImgLocation);
        goldCheckTrainingBitmap.Save(goldCheckTrainImgLocation);
      }
    }
  
  }
}
