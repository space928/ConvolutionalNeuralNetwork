using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.UserInterface;
using OpenCvSharp.Extensions;

namespace NeuralNetwork
{
    public class Convolution
    {
        public List<Mat> srcImgs = new List<Mat>();
        public List<Mat> targetImgs = new List<Mat>();
        public Mat[] kernels;
        //public Mat[] bestKernels;
        public float bestKernelFitness = 0;
        public Mat dstImg = new Mat();
        public Size inSize;

        public Convolution()
        {

        }

        public void LoadImage(string inpath)
        {
            try
            {
                byte[] imageData = File.ReadAllBytes(inpath);
                srcImgs[0] = Mat.FromImageData(imageData, ImreadModes.Color);
            } catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }

            inSize = srcImgs[0].Size();

            dstImg = new Mat(srcImgs[0].Size(), srcImgs[0].Type());
        }

        public void LoadImages(string inPath, string targetPath)
        {
            string[] images = Directory.GetFiles(inPath);
            for (int i = 0; i < images.Length; i++)
            {
                try
                {
                    byte[] imageData = File.ReadAllBytes(images[i]);
                    srcImgs.Add(Mat.FromImageData(imageData, ImreadModes.Color));
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }

            images = Directory.GetFiles(targetPath);
            for (int i = 0; i < images.Length; i++)
            {
                try
                {
                    byte[] imageData = File.ReadAllBytes(images[i]);
                    targetImgs.Add(Mat.FromImageData(imageData, ImreadModes.Color));
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }

            if (srcImgs.Count < 1)
            {
                Console.WriteLine("FATAL: No source images could be loaded!");
                return;
            }

            if (srcImgs.Count != targetImgs.Count)
            {
                Console.WriteLine("FATAL: Number of source images does not match number of target images!");
                return;
            }

            inSize = srcImgs[0].Size();

            dstImg = new Mat(srcImgs[0].Size(), srcImgs[0].Type());
        }

        public void InitialiseKernel(int width, int height, int layers)
        {
            kernels = new Mat[layers];
            for (int i = 0; i<kernels.Length; i++)
            {
                kernels[i] = new Mat(new Size(width, height), MatType.CV_32FC1);
                kernels[i].SetTo(0);
                kernels[i].Set<float>(kernels[i].Rows / 2, kernels[i].Cols / 2, 1);
            }
        }

        public Mat[] MutateKernel(float mutationRate)
        {
            Mat[] outp = new Mat[kernels.Length];
            for (int i = 0; i < kernels.Length; i++)
            {
                outp[i] = new Mat(kernels[i].Size(), MatType.CV_32FC1);
                Mat rnd = new Mat(kernels[i].Size(), MatType.CV_32FC1);
                rnd.Randu(new Scalar(-mutationRate), new Scalar(mutationRate));
                //rnd.Randu(new Scalar(0.0), new Scalar(0.0));
                //This might work
                Cv2.Add(kernels[i], rnd, kernels[i]);
                /*for (int x = 0; x < rnd.Rows; x++)
                {
                    for (int j = 0; j < rnd.Cols; j++)
                    {
                        Console.Write(rnd.At<float>(x, j) + " , ");
                    }
                    Console.WriteLine();
                }
                for (int x = 0; x < rnd.Rows; x++)
                {
                    for (int j = 0; j < rnd.Cols; j++)
                    {
                        Console.Write(kernels[i].At<float>(x, j) + " , ");
                    }
                    Console.WriteLine();
                }*/
                rnd.Dispose();
            }
            return outp;
        }

        public void Convolve(int index, Program.Settings settings)
        {
            dstImg = srcImgs[index].Filter2D(-1, kernels[0]);
            for (int i = 1; i < kernels.Length; i++)
            {
                //dstImg = dstImg.Filter2D(-1, kernels[i]);
                if (inSize.Width != settings.midLayerRes || inSize.Width != settings.outLayerRes)//Assuming width and height are the same
                {
                    //This could be slow...
                    float respos = ((i / ((float)kernels.Length - 1)) * 2);
                    int size = (int)Lerp(Lerp(inSize.Width, settings.midLayerRes, respos), settings.outLayerRes, respos-1f);
                    size = ToNearestPowerOf2(size);
                    Size nSize = new Size(size, size);
                    Cv2.Resize(dstImg, dstImg, nSize);
                }
                Cv2.Filter2D(dstImg, dstImg, -1, kernels[i]);
            }
            //File.WriteAllBytes("tmpimg.png", dstImg.ToBytes());
        }

        public float EvaluateFitness(int index, Program.Settings settings)
        {
            Convolve(index, settings);
            Mat targetTmp = new Mat();
            Mat dstTmp = new Mat();
            targetImgs[index].ConvertTo(targetTmp, MatType.CV_32FC3);
            dstImg.ConvertTo(dstTmp, MatType.CV_32FC3);
            //targetTmp *= -1;

            //Cv2.Add(dstImg, targetTmp, targetTmp, null, MatType.CV_32F);
            Cv2.Absdiff(dstTmp, targetTmp, targetTmp);
            dstTmp.Dispose();

            return -(float)targetTmp.Sum();
        }

        public Mat[] DeepCopyKernels()
        {
            Mat[] outp = new Mat[kernels.Length];
            for(int i = 0; i < kernels.Length; i++)
            {
                outp[i] = new Mat();
                kernels[i].CopyTo(outp[i]);
            }
            return outp;
        }

        public void SetKernels(Mat[] newKernels)
        {
            int i = 0;
            foreach (Mat k in newKernels)
            {
                kernels[i] = k.Clone();
                i++;
            }
        }

        float Lerp(float p1, float p2, float fraction)
        {
            return p1 + (p2 - p1) * Math.Min(Math.Max(fraction, 0f), 1f);
        }

        int ToNearestPowerOf2(int x)
        {
            int next = x;
            --next;
            next |= next >> 1;
            next |= next >> 2;
            next |= next >> 4;
            next |= next >> 8;
            next |= next >> 16;
            next = next + 1;
            int prev = next >> 1;
            return next - x < x - prev ? next : prev;
        }
    }
}
