using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Windows.Forms;
using System.IO;

namespace RoadExtraction
{
  class RoadExtraction
  {
    [STAThread]
    static void Main(string[] args)
    {
      // Network Parameters:
      #region
      int nrOfPixelChannels = 3; // i.e. RGB
      int networkInputPatchDim = 64; // 64 in report, Has to be a multiple of the networkOutputDim.
      int networkOutputPatchDim = 16; // 16 in report
      int nrOfOutputs = networkOutputPatchDim * networkOutputPatchDim;

      int filterDim = 12; // 12 in report
      int nrOfFilters = 80; // 64 in report
      int filterStride = 4; // 4 in report
      int featureMapDim = (networkInputPatchDim - filterDim) / filterStride + 1;

      int[] weightDimForConvLayer = { nrOfFilters, filterDim, filterDim, nrOfPixelChannels }; // W(f,a,b,ff): (a,b,ff) = filter position,  f = feature map index next layer;
      int[] weightDimForOutputLayer = { nrOfOutputs, featureMapDim, featureMapDim, nrOfFilters }; // W(k,i,j,f): (i,j,f) = neuron position input layer,  k = neuron index this layer;

      string activationFunctionConvoluted = "relu"; // relu in report
      string activationFunctionOutput = "sigmoid"; // sigmoid in report
      string lossFunction = "crossentropy"; // crossentropy in report
      #endregion

      // Training parameters:
      #region
      // Due to rotation and edge effects (difference between network input and output dim) the output image will be 
      // smaller. At least by a factor of sqrt(2)/2 due to the rotation
      
      int nrOfTrainingImagePairs, nrOfTrainingImagePairsInMemory, nrOfValidationImagePairs, nrOfImagePairs;
      bool visuallyCheckTraining = true;

      // Termination conditions:
      int maxNrOfEpochs;
      int maxNrOfMinutes;

      int miniBatchSize = 100;
      int calculateErrorInterval = 1;
      int updateMomentumLearningRateInterval = 2;

      double initLearningRate = 2.5 * Math.Pow(10, -6);; // constant 0.0005 in report, 0.000005 worked for single image (841) mini batch size
      double finalLearningRate = 3.125 * Math.Pow(10, -7);
      double initMomentum = 0.6;// constant 0.9 in report
      double finalMomentum = 0.9;
      double weightDecay = 0.0001; // 0.0002 in report
      #endregion
  
      // Other parameters:
      #region
      double roadProbabilityThreshold = 0.5;
      #endregion

      // Main menu:
      #region
      string mainMenuSelection, sel1MenuSelection, sel2MenuSelection, sel2MenuSelection2, sel5MenuSelection, sel5MenuSelection2,
        inputImagePath, goldImagePath, networkFolderLocation, inputImageFolder, saveImageFolder, runAnotherImage;
      OpenFileDialog fileDialog = new OpenFileDialog();
      SaveFileDialog saveDialog = new SaveFileDialog();
      FolderBrowserDialog folderDialog = new FolderBrowserDialog();
      Bitmap roadMap;
      bool displayMenu = true;
      string currentDirectory = Directory.GetCurrentDirectory();
      //System.Media.SoundPlayer goodbyePlayer = new System.Media.SoundPlayer(currentDirectory + "\\Audio\\" + "t2_hasta_la_vista.wav");
      while(displayMenu) {

        // Main header:
        #region
        System.Console.WriteLine("//////    //////        ////       //////      //////// //      // //////////");
        System.Console.WriteLine("//   //  //    //      //  //      //   //     //        //    //      //    ");
        System.Console.WriteLine("//   // //      //    //    //     //    //    //         //  //       //    ");
        System.Console.WriteLine("/////   //      //   //////////    //    //    //////      ////        //    ");
        System.Console.WriteLine("// //   //      //  //        //   //    //    //         //  //       //    ");
        System.Console.WriteLine("//  //   //    //  //          //  //   //     //        //    //      //    ");
        System.Console.WriteLine("//   //   //////  //            // //////      //////// //      //     //    ");
        System.Console.WriteLine("©2015\n");
        #endregion

        mainMenuSelection = Menu.MainMenu();

        switch (mainMenuSelection) {
          case "1":
            // Training a network:
            #region
            sel1MenuSelection = Menu.Sel1Menu();
            
            ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork ANNTrain = 
              new ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork(weightDimForConvLayer, weightDimForOutputLayer, 
                networkInputPatchDim, networkOutputPatchDim);
            // Adding layers to the network: (Note in the current state these two layers has to be added!)
            ANNTrain.InitializeConvolutionalLayer(networkInputPatchDim, nrOfPixelChannels, activationFunctionConvoluted, nrOfFilters, filterDim, filterStride);
            ANNTrain.InitializeFullyConnectedOutputLayer(featureMapDim, nrOfFilters, activationFunctionOutput, networkOutputPatchDim, lossFunction);

            if (sel1MenuSelection == "1") {
              ANNTrain.InitializeRandomWeightAndBiases();
            } else if (sel1MenuSelection == "2") {
              Dialogs.LoadNeuralNetworkOption(ref ANNTrain);
            } else {
              System.Console.WriteLine("Wrong input please try again.");
              System.Console.WriteLine();
              break;
            }
            System.Console.WriteLine("Select network folder...");
            folderDialog.ShowDialog();
            networkFolderLocation = folderDialog.SelectedPath + "\\";
            System.Console.WriteLine("Selected network folder: " + networkFolderLocation);
            System.Console.WriteLine();

            Directory.CreateDirectory(networkFolderLocation + "Training set");
            Directory.CreateDirectory(networkFolderLocation + "Validation set");
            
            System.Console.WriteLine("Put the corresponding image pairs SatX.bmp and GoldX.bmp X = {1,2...,N} in the");
            System.Console.WriteLine("created subfolders \"Training set\" and \"Validation set\" located in the network");
            System.Console.WriteLine("folder location.");
            System.Console.WriteLine("Press any key to continue when this is done...");
            System.Console.ReadKey();
            System.Console.WriteLine();

            // Get number of image pairs
            DirectoryInfo dirInfoTrain = new DirectoryInfo(networkFolderLocation + "Training set");
            DirectoryInfo dirInfoValidation = new DirectoryInfo(networkFolderLocation + "Validation set");
            nrOfTrainingImagePairs = dirInfoTrain.GetFiles().Length/2;
            nrOfValidationImagePairs = dirInfoValidation.GetFiles().Length/2;

            System.Console.Write("Enter number of training image pairs which can be loaded in memory: ");
            nrOfTrainingImagePairsInMemory = Int32.Parse(System.Console.ReadLine());
            
            System.Console.Write("Enter max number of epochs: ");
            maxNrOfEpochs = Int32.Parse(System.Console.ReadLine());
            System.Console.Write("Enter max number of minutes: ");
            maxNrOfMinutes = Int32.Parse(System.Console.ReadLine());
            System.Console.WriteLine();
            System.Console.WriteLine("Training algorithm is running...");
            System.Console.WriteLine("Press \"esc\" to exit and save network.");
            System.Console.WriteLine();

            ANNTrain.TrainNetwork(networkFolderLocation, maxNrOfEpochs, maxNrOfMinutes, miniBatchSize, initLearningRate, 
              finalLearningRate, initMomentum, finalMomentum, weightDecay, calculateErrorInterval, updateMomentumLearningRateInterval, 
              nrOfTrainingImagePairs, nrOfTrainingImagePairsInMemory, nrOfValidationImagePairs, roadProbabilityThreshold, visuallyCheckTraining);
            
              System.Console.WriteLine();
            System.Console.WriteLine("Training complete!");
            System.Console.WriteLine("Network parameters saved in: " + networkFolderLocation + "Network parameters");
            System.Console.WriteLine();
            System.Console.ReadKey();
            #endregion
            break;
          case "2":
            // Run saved neural network
            #region
            ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork ANNRun = 
              new ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork(weightDimForConvLayer, weightDimForOutputLayer, 
                networkInputPatchDim, networkOutputPatchDim);
            // Adding layers to the network: (Note in the current state these two layers has to be added!)
            ANNRun.InitializeConvolutionalLayer(networkInputPatchDim, nrOfPixelChannels, activationFunctionConvoluted, nrOfFilters, filterDim, filterStride);
            ANNRun.InitializeFullyConnectedOutputLayer(featureMapDim, nrOfFilters, activationFunctionOutput, networkOutputPatchDim, lossFunction);

            Dialogs.LoadNeuralNetworkOption(ref ANNRun);
            sel2MenuSelection = Menu.Sel2Menu();
            sel2MenuSelection2 = Menu.Sel2Menu2();

            if ((sel2MenuSelection != "1" && sel2MenuSelection != "2") || (sel2MenuSelection2 != "1" && sel2MenuSelection2 != "2")) {
              System.Console.WriteLine("Wrong input please try again.");
              System.Console.WriteLine();
              break;
            }

            if (sel2MenuSelection2 == "1") {
              runAnotherImage = "1";

              while (runAnotherImage == "1") {
                inputImagePath = Dialogs.SelectInputImage();

                System.Console.WriteLine("Running road detection algorithm...");
                if (sel2MenuSelection == "1") {
                  roadMap = ANNRun.GetRoadBitmap(inputImagePath, roadProbabilityThreshold);
                } else {
                  roadMap = ANNRun.GetRawRoadBitmap(inputImagePath);
                }
                Dialogs.SaveBitmap(roadMap);
                System.Console.WriteLine("Road detection complete!");
                System.Console.WriteLine();

                runAnotherImage = Menu.RunAnotherImage();
              }
            } else {
              System.Console.WriteLine("Put the images named SatX.bmp X = {1,2...,N} in an input image folder");
              inputImageFolder = Dialogs.SelectInputImageFolder();

              DirectoryInfo dirInfo = new DirectoryInfo(inputImageFolder);
              int nrOfImages = dirInfo.GetFiles("Sat*").Length;
              saveImageFolder = Dialogs.SelectSaveImageFolder();

              System.Console.WriteLine("Running road detection algorithm...");
              for (int i = 1; i <= nrOfImages; ++i) {
                if (sel2MenuSelection == "1") {
                  roadMap = ANNRun.GetRoadBitmap(inputImageFolder + "Sat" + i + ".bmp", roadProbabilityThreshold);
                } else {
                  roadMap = ANNRun.GetRawRoadBitmap(inputImageFolder + "Sat" + i + ".bmp");
                }
                roadMap.Save(saveImageFolder + "Sat" + i + ".bmp");
              }
              System.Console.WriteLine("Road detection complete!");
              System.Console.WriteLine();
            }
            System.Console.ReadKey();
            #endregion
            break;
          case "3":
            // Run road extraction algorithm
            #region
            ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork ANNInit = 
              new ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork(weightDimForConvLayer, weightDimForOutputLayer, 
                networkInputPatchDim, networkOutputPatchDim);
            // Adding layers to the network: (Note in the current state these two layers has to be added!)
            ANNInit.InitializeConvolutionalLayer(networkInputPatchDim, nrOfPixelChannels, activationFunctionConvoluted, nrOfFilters, filterDim, filterStride);
            ANNInit.InitializeFullyConnectedOutputLayer(featureMapDim, nrOfFilters, activationFunctionOutput, networkOutputPatchDim, lossFunction);

            ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork ANNStructure = 
              new ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork(weightDimForConvLayer, weightDimForOutputLayer, 
                networkInputPatchDim, networkOutputPatchDim);
            // Adding layers to the network: (Note in the current state these two layers has to be added!)
            ANNStructure.InitializeConvolutionalLayer(networkInputPatchDim, nrOfPixelChannels, activationFunctionConvoluted, nrOfFilters, filterDim, filterStride);
            ANNStructure.InitializeFullyConnectedOutputLayer(featureMapDim, nrOfFilters, activationFunctionOutput, networkOutputPatchDim, lossFunction);

            Dialogs.LoadNeuralNetworkOption(ref ANNInit);
            Dialogs.LoadNeuralNetworkOption(ref ANNStructure);

            runAnotherImage = "1";

            while (runAnotherImage == "1") {
              inputImagePath = Dialogs.SelectInputImage();

              System.Console.WriteLine("Running road extraction algorithm...");
              Bitmap rawRoadMap = ANNInit.GetRawRoadBitmap(inputImagePath);
              roadMap = ANNStructure.GetRoadBitmap(rawRoadMap, roadProbabilityThreshold);
              System.Console.WriteLine("Road extraction complete!");
              System.Console.WriteLine();
              Dialogs.SaveBitmap(roadMap);
              System.Console.ReadKey();
    
              runAnotherImage = Menu.RunAnotherImage();
            }
            #endregion
            break;
          case "4":
            // Testing scripts:
            #region
            ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork ANNTest = 
              new ArtificialNeuralNetworkLibrary.ArtificialNeuralNetwork(weightDimForConvLayer, weightDimForOutputLayer, 
                networkInputPatchDim, networkOutputPatchDim);
            // Adding layers to the network: (Note in the current state these two layers has to be added!)
            ANNTest.InitializeConvolutionalLayer(networkInputPatchDim, nrOfPixelChannels, activationFunctionConvoluted, nrOfFilters, filterDim, filterStride);
            ANNTest.InitializeFullyConnectedOutputLayer(featureMapDim, nrOfFilters, activationFunctionOutput, networkOutputPatchDim, lossFunction);

            System.Console.WriteLine("Testing the backpropagation of the network using finite difference calculus on \none random input patch.");
            System.Console.WriteLine("A relative error less then 1e-04 is good as epsilon gets smaller.");
            System.Console.WriteLine();

            inputImagePath = Dialogs.SelectInputImage();
            goldImagePath = Dialogs.SelectGoldImage();

            ANNTest.InitializeRandomWeightAndBiases();
            ANNTest.TestNetwork(inputImagePath, goldImagePath);

            System.Console.WriteLine("Testing complete!");
            System.Console.WriteLine();
            System.Console.ReadKey();
            #endregion
            break;
          case "5":
            //Data processing:
            #region
            sel5MenuSelection = Menu.Sel5Menu();

            if (sel5MenuSelection == "1") {
              // Preprocess google images
              #region
              System.Console.WriteLine("Put the Google image pairs named SatX.bmp and GoldX.bmp X = {1,2...,N}");
              System.Console.WriteLine("in an input folder.");
              System.Console.WriteLine("Press any key to continue when this is done...");
              System.Console.ReadKey();
              System.Console.WriteLine();

              inputImageFolder = Dialogs.SelectInputImageFolder();
              saveImageFolder = Dialogs.SelectSaveImageFolder();

              DirectoryInfo dirInfo = new DirectoryInfo(inputImageFolder);
              nrOfImagePairs = dirInfo.GetFiles().Length / 2;

              System.Console.Write("Enter number of boarder pixels to remove for the Google logo: ");
              int googleLogoCropSize = Int32.Parse(System.Console.ReadLine());
              System.Console.Write("Do you want to randomly rotate the images (y/n): ");
              bool rotateImages = (System.Console.ReadLine() == "y") ? true : false;

              MiscLibrary.ImageProcessor.PreprocessGoogleMapImages(inputImageFolder, saveImageFolder, nrOfImagePairs,
                googleLogoCropSize, networkOutputPatchDim, rotateImages);

              System.Console.WriteLine("Preprocessing complete!");
              System.Console.WriteLine();
              #endregion
            } else if (sel5MenuSelection == "2") {
              // Crop images
              #region

              sel5MenuSelection2 = Menu.Sel5Menu2();

              System.Console.Write("Enter crop height: ");
              int cropWidth = Int32.Parse(System.Console.ReadLine());
              System.Console.Write("Enter crop width: ");
              int cropHeight = Int32.Parse(System.Console.ReadLine());

              if (sel5MenuSelection2 == "1") {
                runAnotherImage = "1";

                while (runAnotherImage == "1") {
                  inputImagePath = Dialogs.SelectInputImage();
                  Bitmap cropBitmap = new Bitmap(inputImagePath);
                  cropBitmap = MiscLibrary.ImageProcessor.CropBitmap(cropBitmap, cropWidth, cropHeight);
                  Dialogs.SaveBitmap(cropBitmap);

                  System.Console.WriteLine("Cropping complete!");
                  System.Console.WriteLine();

                  runAnotherImage = Menu.RunAnotherImage();
                }
                
              } else if (sel5MenuSelection2 == "2") {
                System.Console.Write("Enter image prefixes: ");
                string imgPrefix = System.Console.ReadLine();

                System.Console.WriteLine("Put the images that are to be cropped in an input image folder.");
                System.Console.WriteLine("Name the images as " + imgPrefix + "X.bmp where X = {1,2...,N}.");

                inputImageFolder = Dialogs.SelectInputImageFolder();
                DirectoryInfo dirInfo = new DirectoryInfo(inputImageFolder);
                int nrOfImages = dirInfo.GetFiles(imgPrefix + "*").Length;
                saveImageFolder = Dialogs.SelectSaveImageFolder();

                for (int i = 1; i <= nrOfImages; ++i) {
                  Bitmap cropBitmap = new Bitmap(inputImageFolder + imgPrefix + i + ".bmp");
                  cropBitmap = MiscLibrary.ImageProcessor.CropBitmap(cropBitmap, cropWidth, cropHeight);
                  cropBitmap.Save(saveImageFolder + imgPrefix + i + ".bmp");
                }
                System.Console.WriteLine("Cropping complete!");
                System.Console.WriteLine();
              } else {
                System.Console.WriteLine("Wrong menu input please try again.");
                System.Console.WriteLine();
              }

              #endregion
            } else if (sel5MenuSelection == "3") {
              // Rename bitmaps
              #region
              System.Console.Write("Enter image prefixes: ");
              string imgPrefix = System.Console.ReadLine();
              System.Console.WriteLine("Put the images that are to be renamed in an input image folder.");
              inputImageFolder = Dialogs.SelectInputImageFolder();
              saveImageFolder = Dialogs.SelectSaveImageFolder();
              System.Console.WriteLine("The images will be named as " + imgPrefix + "X.bmp where X = {1,2...,N}");

              DirectoryInfo dirInfo = new DirectoryInfo(inputImageFolder);
              FileInfo[] fileInfo = dirInfo.GetFiles(imgPrefix + "*");
              int counter = 1;
              foreach(FileInfo f in fileInfo) {
                File.Move(f.FullName, saveImageFolder + imgPrefix + counter + ".bmp");
                ++counter;
              }
              System.Console.WriteLine("Bitmap renaming complete!");
              #endregion
            } else {
              System.Console.WriteLine("Wrong input please try again.");
              System.Console.WriteLine();
            }
            System.Console.ReadKey();
            #endregion
            break;
          case "6":
            //Evaluate road map
            #region
            inputImagePath = Dialogs.SelectInputImage();
            goldImagePath = Dialogs.SelectGoldImage();
            Bitmap inputImage = new Bitmap(inputImagePath);

            int classificationError = MiscLibrary.ImageProcessor.CalculateClassificationError(inputImagePath, goldImagePath);
            int[] evaluationArray = MiscLibrary.ImageProcessor.EvaluateRoadMap(inputImagePath, goldImagePath);
            // evaluationArray = {totNrOfRoadPixels, correctRoadPixels, incorrectRoadPixels, 
            //  totNrOfNonRoadPixels, correctNonRoadPixels, incorrectNonRoadPixels}

            System.Console.WriteLine("Evaluation:");
            System.Console.WriteLine("Total number of pixels: {0}", evaluationArray[0] + evaluationArray[3]);
            System.Console.WriteLine("Classification error: {0}, {1}%", classificationError, 100 * (double)classificationError / (evaluationArray[0] + evaluationArray[3]));
            System.Console.WriteLine("Total number of road pixels: {0}", evaluationArray[0]);
            System.Console.WriteLine("Correct classified road pixels: {0}, {1}%", evaluationArray[1], 100 * (double)evaluationArray[1] / evaluationArray[0]);
            System.Console.WriteLine("Incorrect classified road pixels: {0}", evaluationArray[2]);
            System.Console.WriteLine("Total number of non road pixels: {0}", evaluationArray[3]);
            System.Console.WriteLine("Correct classified non road pixels: {0}, {1}%", evaluationArray[4], 100 * (double)evaluationArray[4] / evaluationArray[3]);
            System.Console.WriteLine("Incorrect classified non road pixels: {0}%", evaluationArray[5]);
            System.Console.WriteLine();
            double precision = (double)evaluationArray[1] / (evaluationArray[1] + evaluationArray[5]);
            double recall = (double)evaluationArray[1] / (evaluationArray[1] + evaluationArray[2]);
            double quality = (double)evaluationArray[1] / (evaluationArray[1] + evaluationArray[2] + evaluationArray[5]);
            double accuracy = (double)(evaluationArray[1] + evaluationArray[4]) / (evaluationArray[0] + evaluationArray[3]);
            System.Console.WriteLine("Precision: {0}", precision);
            System.Console.WriteLine("Recall: {0}", recall);
            System.Console.WriteLine("Quality: {0}", quality);
            System.Console.WriteLine("Accuracy: {0}", accuracy);
            System.Console.WriteLine("Evaluation complete!");
            System.Console.ReadKey();
            #endregion
            break;
          case "7":
            // Bye Bye
            #region
            System.Console.WriteLine("Hasta la vista, baby!");
            //goodbyePlayer.Play();
            System.Console.ReadKey();
            #endregion
            return;
          default:
            // Input error
            #region
            System.Console.WriteLine("Wrong input, please try again.");
            System.Console.WriteLine();
            System.Console.ReadKey();
            #endregion
            break;
        }
      }
      #endregion

    }
  }
}
