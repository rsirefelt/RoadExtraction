using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;

namespace MiscLibrary
{
  public unsafe static class ImageProcessor
  {
    // Constants:
    #region
    private const byte PIXEL_SIZE = 3;
    private const byte MAXIMUM_PIXEL_VALUE = 255;
    private const byte NR_OF_PIXEL_CHANNELS = 3;
    private const byte RED_CHANNEL = 0, BLUE_CHANNEL = 1, GREEN_CHANNEL = 2;
    private const byte GREY_SCALE_SIZE = 2;
    #endregion


    /// <summary>
    /// Gets RGB pixel values of from the bitmap (.bmp)
    /// </summary>
    /// <param name="bitmapLocation"></param>
    /// <returns></returns>
    public static byte[][][] GetPixelValues(Bitmap bitmap)
    {
      byte[][][] pixelValues;

      int width = bitmap.Width;
      int height = bitmap.Height;

      pixelValues = MiscLibrary.Initialize.AllocateByte(width, height, NR_OF_PIXEL_CHANNELS);

      BitmapData bitmapData = bitmap.LockBits(
                 new Rectangle(0, 0, width, height),
                 ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

      byte* pRow;
      for (int i = 0; i < height; ++i) {
        pRow = (byte*)bitmapData.Scan0 + (i * bitmapData.Stride);
        for (int j = 0; j < width; ++j) {
          pixelValues[i][j][RED_CHANNEL] = (byte)(pRow[j * PIXEL_SIZE + 2]);
          pixelValues[i][j][GREEN_CHANNEL] = (byte)(pRow[j * PIXEL_SIZE + 1]);
          pixelValues[i][j][BLUE_CHANNEL] = (byte)(pRow[j * PIXEL_SIZE]);
        }
      }
      bitmap.UnlockBits(bitmapData);

      return pixelValues;
    }


    /// <summary>
    /// Gets RGB pixel values of from the bitmap (.bmp) file located at bitmapLocation
    /// </summary>
    /// <param name="bitmapLocation"></param>
    /// <returns></returns>
    public static byte[][][] GetPixelValues(string bitmapLocation)
    {
      byte[][][] pixelValues;

      Bitmap bitmap = (Bitmap)Image.FromFile(bitmapLocation);

      int width = bitmap.Width;
      int height = bitmap.Height;

      pixelValues = MiscLibrary.Initialize.AllocateByte(width, height, NR_OF_PIXEL_CHANNELS);

      BitmapData bitmapData = bitmap.LockBits(
                 new Rectangle(0, 0, width, height),
                 ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

      byte* pRow;
      for (int i = 0; i < height; ++i) {
        pRow = (byte*)bitmapData.Scan0 + (i * bitmapData.Stride);
        for (int j = 0; j < width; ++j) {
          pixelValues[i][j][RED_CHANNEL] = (byte)(pRow[j * PIXEL_SIZE + 2]);
          pixelValues[i][j][GREEN_CHANNEL] = (byte)(pRow[j * PIXEL_SIZE + 1]);
          pixelValues[i][j][BLUE_CHANNEL] = (byte)(pRow[j * PIXEL_SIZE]);
        }
      }

      bitmap.UnlockBits(bitmapData);

      return pixelValues;
    }


    /// <summary>
    /// Returns the bitmap from the inserted pixelValues
    /// </summary>
    /// <param name="pixelValues"></param>
    /// <returns></returns>
    public static Bitmap GetBitmap(byte[][][] pixelValues)
    {
      int height = pixelValues.GetLength(0);
      int width = pixelValues[0].GetLength(0);

      Bitmap bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb);
     
      BitmapData bitmapData = bitmap.LockBits(
                 new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                 ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
      byte* pRow;
      for (int i = 0; i < height; i++) {
        pRow = (byte*)bitmapData.Scan0 + (i * bitmapData.Stride);
        for (int j = 0; j < width; j++) {
          pRow[j * PIXEL_SIZE + 2] = pixelValues[i][j][RED_CHANNEL];
          pRow[j * PIXEL_SIZE + 1] = pixelValues[i][j][GREEN_CHANNEL];
          pRow[j * PIXEL_SIZE] = pixelValues[i][j][BLUE_CHANNEL];
        }
      }
      bitmap.UnlockBits(bitmapData);

      return bitmap;
    }


    /// <summary>
    /// Converts the pixel values to greyscale
    /// </summary>
    /// <param name="rFraction"></param>
    /// <param name="gFraction"></param>
    /// <param name="bFraction"></param>
    /// <param name="pixelValues"></param>
    public static void ConvertToGrayScale(double rFraction, double gFraction, double bFraction, byte[][][] pixelValues)
    {
      int height = pixelValues.GetLength(0);
      int width = pixelValues[0].GetLength(0);

      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          byte grayValue = (byte)Math.Round(rFraction * pixelValues[i][j][RED_CHANNEL] + gFraction * pixelValues[i][j][GREEN_CHANNEL] + bFraction * pixelValues[i][j][BLUE_CHANNEL]);
          pixelValues[i][j][RED_CHANNEL] = grayValue;
          pixelValues[i][j][GREEN_CHANNEL] = grayValue;
          pixelValues[i][j][BLUE_CHANNEL] = grayValue;
        }
      }
    }


    /// <summary>
    /// Converts the image to standard grey scale with preset fraction paramters
    /// </summary>
    /// <param name="pixelValues"></param>
    public static void ConvertToStandardGrayScale(byte[][][] pixelValues)
    {
      ConvertToGrayScale(0.300, 0.590, 0.110, pixelValues);
    }


    /// <summary>
    /// Creates a binary version of the image. This method assumes that the image is in grayscale
    /// </summary>
    /// <param name="threshold"></param>
    /// <param name="pixelValues"></param>
    public static void Binarize(byte threshold, byte[][][] pixelValues)
    {
      int height = pixelValues.GetLength(0);
      int width = pixelValues[0].GetLength(0);

      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          byte grayValue = pixelValues[i][j][RED_CHANNEL];
          grayValue = (grayValue < threshold) ? (byte) 0 : MAXIMUM_PIXEL_VALUE;
          pixelValues[i][j][RED_CHANNEL] = grayValue;
          pixelValues[i][j][GREEN_CHANNEL] = grayValue;
          pixelValues[i][j][BLUE_CHANNEL] = grayValue;
        }
      }
    }


    /// <summary>
    /// Creates a classification version of the image i.e. road pixels get value 1 and non road pixels get value 0.
    /// This method assumes that the image is in grayscale
    /// </summary>
    /// <param name="threshold"></param>
    /// <param name="pixelValues"></param>
    public static void GetRoadPixelClassificationData(byte threshold, byte[][][] pixelValues)
    {
      int height = pixelValues.GetLength(0);
      int width = pixelValues[0].GetLength(0);

      byte nonRoadPixelValue = 0;
      byte roadPixelValue = 1;

      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          byte grayValue = pixelValues[i][j][RED_CHANNEL];
          grayValue = (grayValue < threshold) ? nonRoadPixelValue : roadPixelValue;
          pixelValues[i][j][RED_CHANNEL] = grayValue;
          pixelValues[i][j][GREEN_CHANNEL] = grayValue;
          pixelValues[i][j][BLUE_CHANNEL] = grayValue;
        }
      }
    }


    /// <summary>
    /// Reverts a classification version of the image i.e. road pixels get value 255 and non road pixels get value 0, 
    /// This method assumes that the image is in grayscale
    /// </summary>
    /// <param name="threshold"></param>
    public static void RevertRoadPixelClassificationDataToBinaryData(byte[][][] pixelValues)
    {
      int height = pixelValues.GetLength(0);
      int width = pixelValues[0].GetLength(0);

      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          pixelValues[i][j][RED_CHANNEL] *= MAXIMUM_PIXEL_VALUE;
          pixelValues[i][j][GREEN_CHANNEL] *= MAXIMUM_PIXEL_VALUE;
          pixelValues[i][j][BLUE_CHANNEL] *= MAXIMUM_PIXEL_VALUE;
        }
      }
    }


    /// <summary>
    /// Rotates a bitmap around its center
    /// </summary>
    /// <param name="bitmap"></param>
    /// <param name="angle"></param>
    /// <returns></returns>
    public static Bitmap RotateBitmap(Bitmap bitmap, float angle)
    {
      Bitmap rotatedBitmap = new Bitmap(bitmap.Width, bitmap.Height);
      Graphics graphics = Graphics.FromImage(rotatedBitmap);
      graphics.TranslateTransform((float)bitmap.Width / 2, (float)bitmap.Height / 2);
      graphics.RotateTransform(angle);
      graphics.TranslateTransform(-(float)bitmap.Width / 2, -(float)bitmap.Height / 2);
      graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
      graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
      graphics.DrawImage(bitmap, new Point(0, 0));

      bitmap.Dispose();
      graphics.Dispose();

      return rotatedBitmap;
    }


    /// <summary>
    /// Croppes the center rectangle out of a bitmap
    /// </summary>
    /// <param name="bitmap"></param>
    /// <param name="cropWidth"></param>
    /// <param name="cropHeight"></param>
    /// <returns></returns>
    public static Bitmap CropBitmap(Bitmap bitmap, int cropWidth, int cropHeight)
    {
      Rectangle cropRectangle = new Rectangle(bitmap.Width/2 - cropWidth/2, bitmap.Height/2 - cropHeight/2, cropWidth, cropHeight);
      Bitmap croppedBitmap = bitmap.Clone(cropRectangle, bitmap.PixelFormat);

      bitmap.Dispose();
      return croppedBitmap;
    }


    /// <summary>
    /// Extracts an input image patch. The image patch must be a square.
    /// </summary>
    /// <param name="inputImage"></param>
    /// <param name="startIndRow"></param>
    /// <param name="endIndRow"></param>
    /// <param name="startIndCol"></param>
    /// <returns></returns>
    public static double[][][] GetInputImagePatch(byte[][][] inputImage, int startIndRow, int endIndRow, int startIndCol)
    {
      int patchDim = endIndRow - startIndRow + 1;
      int nrOfChannels = 3; // RGB
      int inputRowInd, inputColInd;
      
      double[][][] imagePatch = MiscLibrary.Initialize.AllocateDouble(patchDim, patchDim, nrOfChannels);

      for (int i = 0; i < patchDim; ++i) {
        inputRowInd = i + startIndRow;
        for (int j = 0; j < patchDim; ++j) {
          inputColInd = j + startIndCol;

          imagePatch[i][j][0] = inputImage[inputRowInd][inputColInd][RED_CHANNEL];
          imagePatch[i][j][1] = inputImage[inputRowInd][inputColInd][GREEN_CHANNEL];
          imagePatch[i][j][2] = inputImage[inputRowInd][inputColInd][BLUE_CHANNEL];
        }
      }

      return imagePatch;
    }

    
    /// <summary>
    /// Extracts an output image patch as a raster vector. The reason for this is so that the network is more easily constructed.
    /// The image patch must be a square and in binary grayscale
    /// </summary>
    /// <param name="roadImage"></param>
    /// <param name="startIndRow"></param>
    /// <param name="endIndRow"></param>
    /// <param name="startIndCol"></param>
    /// <returns></returns>
    public static byte[] GetOutputImagePatch(byte[][][] roadImage, int startIndRow, int endIndRow, int startIndCol)
    {
      int patchDim = endIndRow - startIndRow + 1;
      int imageRasterLength = patchDim * patchDim;
      int inputRowInd, inputColInd;

      byte[] roadImagePatch = new byte[imageRasterLength];

      for (int i = 0; i < patchDim; ++i) {
        inputRowInd = i + startIndRow;
        for (int j = 0; j < patchDim; ++j) {
          inputColInd = j + startIndCol;

          roadImagePatch[i * patchDim + j] = roadImage[inputRowInd][inputColInd][0]; // All pixel channels should be the same
        }
      }

      return roadImagePatch;
    }


    /// <summary>
    /// Preprocess google map images to training data. This is done by cropping out the google logo, 
    /// cropp the images to appropriate size based on 45 degree rotations and optionally randomly rotates 
    /// the images to avoid any unwanted direction artifacts in the neural network.
    /// </summary>
    /// <param name="inputImageSetFolder"></param>
    /// <param name="outputImageSetFolder"></param>
    /// <param name="nrOfImagePairs"></param>
    /// <param name="initialGoogleLogoCropSize"></param>
    /// <param name="networkOutputDim"></param>
    /// <param name="rotateImages"></param>
    public static void PreprocessGoogleMapImages(string inputImageSetFolder, string outputImageSetFolder, int nrOfImagePairs, 
      int initialGoogleLogoCropSize, int networkOutputDim, bool rotateImages)
    {
      Random rng = new Random();
      string satImgStr = "Sat", goldImgStr = "Gold", imgExt = ".bmp";
      string satImgLocation, goldImgLocation, satImgSaveLocation, goldImgSaveLocation;
      Bitmap satBitmap, goldBitmap, satRotatedBitmap, goldRotatedBitmap, satRotatedCroppedBitmap, goldRotatedCroppedBitmap;
      Bitmap sizeBitmap = new Bitmap(inputImageSetFolder + satImgStr + "1" + imgExt);
      int imageHeight = sizeBitmap.Height - 2 * initialGoogleLogoCropSize;
      int imageWidth = sizeBitmap.Width - 2 * initialGoogleLogoCropSize;
      
      int imageOutputCropHeight = (int)(Math.Sqrt(2) * imageHeight / 2);
      imageOutputCropHeight = (imageOutputCropHeight / networkOutputDim) * networkOutputDim;
      int imageOutputCropWidth = (int)(Math.Sqrt(2) * imageWidth / 2);
      imageOutputCropWidth = (imageOutputCropWidth / networkOutputDim) * networkOutputDim;

      for (int imageInd = 0; imageInd < nrOfImagePairs; ++imageInd) {
        satImgLocation = inputImageSetFolder + satImgStr + (imageInd + 1).ToString() + imgExt;
        satImgSaveLocation = outputImageSetFolder + satImgStr + (imageInd + 1).ToString() + imgExt;
        goldImgLocation = inputImageSetFolder + goldImgStr + (imageInd + 1).ToString() + imgExt;
        goldImgSaveLocation = outputImageSetFolder + goldImgStr + (imageInd + 1).ToString() + imgExt;
        satBitmap = new Bitmap(satImgLocation);
        goldBitmap = new Bitmap(goldImgLocation);

        // Remove google logo
        satBitmap = MiscLibrary.ImageProcessor.CropBitmap(satBitmap, imageWidth, imageHeight);
        goldBitmap = MiscLibrary.ImageProcessor.CropBitmap(goldBitmap, imageWidth, imageHeight);

        float rotationAngle = 0;
        if (rotateImages)
          rotationAngle = (float)rng.NextDouble() * 360;

        satRotatedBitmap = satBitmap;
        satRotatedBitmap = MiscLibrary.ImageProcessor.RotateBitmap(satBitmap, rotationAngle);
        satRotatedCroppedBitmap = MiscLibrary.ImageProcessor.CropBitmap(satRotatedBitmap, imageOutputCropWidth, imageOutputCropHeight);
        goldRotatedBitmap = goldBitmap;
        goldRotatedBitmap = MiscLibrary.ImageProcessor.RotateBitmap(goldBitmap, rotationAngle);
        goldRotatedCroppedBitmap = MiscLibrary.ImageProcessor.CropBitmap(goldRotatedBitmap, imageOutputCropWidth, imageOutputCropHeight);

        satRotatedCroppedBitmap.Save(satImgSaveLocation);
        goldRotatedCroppedBitmap.Save(goldImgSaveLocation);
      }      
    }


    /// <summary>
    /// Calculate the road maps classification error
    /// </summary>
    /// <param name="roadMapPath"></param>
    /// <param name="goldImgPath"></param>
    /// <returns></returns>
    public static int CalculateClassificationError(string roadMapPath, string goldImgPath)
    {
      int classificationError = 0;
      byte binaryThreshold = 20;

      Bitmap roadBitmap = new Bitmap(roadMapPath);
      Bitmap goldBitmap = new Bitmap(goldImgPath);

      goldBitmap = CropBitmap(goldBitmap, roadBitmap.Width, roadBitmap.Height);

      byte[][][] roadMapPixelValue = MiscLibrary.ImageProcessor.GetPixelValues(roadBitmap);
      MiscLibrary.ImageProcessor.GetRoadPixelClassificationData(binaryThreshold, roadMapPixelValue);
      byte[][][] goldPixelValues = MiscLibrary.ImageProcessor.GetPixelValues(goldBitmap);
      MiscLibrary.ImageProcessor.ConvertToStandardGrayScale(goldPixelValues);
      MiscLibrary.ImageProcessor.GetRoadPixelClassificationData(binaryThreshold, goldPixelValues);

      for (int i = 0; i < roadBitmap.Height; ++i) {
        for (int j = 0; j < roadBitmap.Width; ++j) {
          classificationError += roadMapPixelValue[i][j][0] * goldPixelValues[i][j][0];
        }
      }

      return classificationError;
    }


    /// <summary>
    /// Returns an evaluation array in the form of
    /// {totNrOfRoadPixels, correctRoadPixels, incorrectRoadPixels, totNrOfNonRoadPixels, correctNonRoadPixels, incorrectNonRoadPixels}
    /// </summary>
    /// <param name="roadMapPath"></param>
    /// <param name="goldImgPath"></param>
    /// <returns></returns>
    public static int[] EvaluateRoadMap(string roadMapPath, string goldImgPath)
    {
      int totNrOfRoadPixels = 0, totNrOfNonRoadPixels = 0;
      int correctRoadPixels = 0, incorrectRoadPixels = 0, correctNonRoadPixels = 0, incorrectNonRoadPixels = 0;
      byte binaryThreshold = 20;

      Bitmap roadBitmap = new Bitmap(roadMapPath);
      Bitmap goldBitmap = new Bitmap(goldImgPath);

      goldBitmap = CropBitmap(goldBitmap, roadBitmap.Width, roadBitmap.Height);

      byte[][][] roadMapPixelValue = MiscLibrary.ImageProcessor.GetPixelValues(roadBitmap);
      MiscLibrary.ImageProcessor.GetRoadPixelClassificationData(binaryThreshold, roadMapPixelValue);
      byte[][][] goldPixelValues = MiscLibrary.ImageProcessor.GetPixelValues(goldBitmap);
      MiscLibrary.ImageProcessor.ConvertToStandardGrayScale(goldPixelValues);
      MiscLibrary.ImageProcessor.GetRoadPixelClassificationData(binaryThreshold, goldPixelValues);

      for (int i = 0; i < roadBitmap.Height; ++i) {
        for (int j = 0; j < roadBitmap.Width; ++j) {
          if (goldPixelValues[i][j][0] == 1) { // Road pixel
            ++totNrOfRoadPixels;
            correctRoadPixels += roadMapPixelValue[i][j][0];
            incorrectRoadPixels += (1 - roadMapPixelValue[i][j][0]);
          } else { // Non road pixel
            ++totNrOfNonRoadPixels;
            correctNonRoadPixels += (1 - roadMapPixelValue[i][j][0]);
            incorrectNonRoadPixels += roadMapPixelValue[i][j][0];
          }
        }
      }

      int[] evaluationArray = { totNrOfRoadPixels, correctRoadPixels, incorrectRoadPixels, 
                                totNrOfNonRoadPixels, correctNonRoadPixels, incorrectNonRoadPixels };
      return evaluationArray;
    }

  }
}
