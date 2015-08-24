using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Drawing;


namespace RoadExtraction
{
  class Dialogs
  {
    /// <summary>
    /// Load network file
    /// </summary>
    /// <param name="ANN"></param>
    static public void LoadNeuralNetworkOption(ref ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork ANN)
    {
      OpenFileDialog fileDialog = new OpenFileDialog();

      fileDialog.Title = "Select a neural network file...";
      System.Console.WriteLine("Select a neural network file...");
      fileDialog.ShowDialog();
      string neuralNetworkFile = fileDialog.FileName;
      System.Console.WriteLine("Selected neural network file: " + neuralNetworkFile);
      System.Console.WriteLine();

      ANN.LoadNetworkParameters(neuralNetworkFile);
    }


    /// <summary>
    /// Select input image dialog
    /// </summary>
    /// <returns></returns>
    static public string SelectInputImage()
    {
      OpenFileDialog fileDialog = new OpenFileDialog();

      fileDialog.Title = "Select input image...";
      System.Console.WriteLine("Select input image...");
      fileDialog.ShowDialog();
      string inputImagePath = fileDialog.FileName;
      System.Console.WriteLine("Selected input image: " + inputImagePath);
      System.Console.WriteLine();

      return inputImagePath;
    }


    /// <summary>
    /// Select gold image dialog
    /// </summary>
    /// <returns></returns>
    static public string SelectGoldImage()
    {
      OpenFileDialog fileDialog = new OpenFileDialog();

      fileDialog.Title = "Select gold image file...";
      System.Console.WriteLine("Select gold image file...");
      fileDialog.ShowDialog();
      string goldImagePath = fileDialog.FileName;
      System.Console.WriteLine("Selected gold image file: " + goldImagePath);
      System.Console.WriteLine();

      return goldImagePath;
    }


    /// <summary>
    /// Select input image folder dialog
    /// </summary>
    /// <returns></returns>
    static public string SelectInputImageFolder()
    {
      FolderBrowserDialog folderDialog = new FolderBrowserDialog();

      System.Console.WriteLine("Select the input image folder...");
      folderDialog.ShowDialog();
      string inputImageFolder = folderDialog.SelectedPath + "\\";
      System.Console.WriteLine("Selected input image folder: " + inputImageFolder);
      System.Console.WriteLine();

      return inputImageFolder;
    }


    /// <summary>
    /// Save road map dialog
    /// </summary>
    /// <param name="roadMap"></param>
    static public void SaveBitmap(Bitmap roadMap)
    {
      SaveFileDialog saveDialog = new SaveFileDialog();

      saveDialog.Title = "Save bitmap as...";
      System.Console.WriteLine("Save bitmap as...");
      saveDialog.ShowDialog();
      System.IO.FileStream fileStream = (System.IO.FileStream)saveDialog.OpenFile();
      roadMap.Save(fileStream, System.Drawing.Imaging.ImageFormat.Bmp);
      System.Console.WriteLine("Bitmap saved as: " + saveDialog.FileName);
      System.Console.WriteLine();

      fileStream.Close();
    }


    /// <summary>
    /// Select image folder dialog
    /// </summary>
    /// <returns></returns>
    static public string SelectSaveImageFolder()
    {
      FolderBrowserDialog folderDialog = new FolderBrowserDialog();

      System.Console.WriteLine("Select the save image folder...");
      folderDialog.ShowDialog();
      string saveImageFolder = folderDialog.SelectedPath + "\\";
      System.Console.WriteLine("Selected save image folder: " + saveImageFolder);
      System.Console.WriteLine();

      return saveImageFolder;
    }

  }
}
