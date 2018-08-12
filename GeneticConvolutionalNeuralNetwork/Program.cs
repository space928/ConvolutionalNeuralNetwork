using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using OpenCvSharp;
using Newtonsoft.Json.Serialization;

namespace NeuralNetwork
{
    public class Program
    {
        public long iters = 0;
        public long gens = 0;
        public long successIters = 0;
        public bool shouldPause = false;
        public DateTime startTime;
        public float currAnneal = 0.0f;
        public Queue<float> runningAvg = new Queue<float>();
        public Parents currentParents = new Parents();
        public Settings settings = new Settings();
        public Convolution convolution;
        public int lastIndex = 0;

        //In the end only one parent is used
        /*public class Parents
        {
            public Mat[] parentKernels;
            public float parentFitness;
            public Parents()
            {
                this.parentKernels = null;
                this.parentFitness = float.MinValue;
            }
            public Parents(Mat[] parentConnections, float parentFitness)
            {
                this.parentKernels = parentConnections;
                this.parentFitness = parentFitness;
            }
            public Parents(Parents source)
            {
                if (source.parentKernels != null)
                    this.parentKernels = source.parentKernels.Clone() as Mat[];//TODO: does this need to be a deep copy?
                else
                    this.parentKernels = new Mat[] { null };
                this.parentFitness = source.parentFitness;
            }
        }*/
        [Serializable]
        public class Parents
        {
            [Serializable]
            public struct Parent
            {
                public Mat[] kernels;
                public float fitness;
                public string species;
                public Parent(float fitness, Mat[] kernels, string species)
                {
                    this.kernels = kernels;
                    this.fitness = fitness;
                    this.species = species;
                }
            }
            public List<Parent> parentKernels;
            public Parents()
            {
                this.parentKernels = new List<Parent>();
            }
            public Parents(Mat[] parentConnections, float parentFitness, string species)
            {
                this.parentKernels = new List<Parent>();
                this.parentKernels.Add(new Parent(parentFitness, parentConnections, species));
            }
            public Parents(Parents source)
            {
                this.parentKernels = new List<Parent>();
                foreach (Parent parentk in source.parentKernels)
                {
                    this.parentKernels.Add(new Parent(parentk.fitness, parentk.kernels.Clone() as Mat[], parentk.species));//TODO: does this need to be a deep copy?
                }
            }
        }

        public class Settings
        {
            public bool highPrecision;
            public bool useRandomDataset;
            public int nodeLayers;
            public int nodesPL;
            public int fitnessAverageIters;
            public float mutationRate;
            public int childrenPerGen;
            public int overrallFitIters;
            public int midLayerRes;
            public int outLayerRes;
            public int parentsPerGeneration;
            public Settings(bool hp, bool rndDat, int nl, int npl, int fai, float mr, int cpg, int ofi, int mlr, int olr, int ppg)
            {
                highPrecision = hp;
                useRandomDataset = rndDat;
                nodeLayers = nl;
                nodesPL = npl;
                fitnessAverageIters = fai;
                mutationRate = mr;
                childrenPerGen = cpg;
                overrallFitIters = ofi;
                midLayerRes = mlr;
                outLayerRes = olr;
                parentsPerGeneration = ppg;
            }
            public Settings()
            {
                highPrecision = false;
                useRandomDataset = false;
                nodeLayers = 6;
                nodesPL = 8;
                fitnessAverageIters = 400;
                mutationRate = 1.5f;
                childrenPerGen = 100;
                overrallFitIters = 2000;
                midLayerRes = 128;
                outLayerRes = 128;
                parentsPerGeneration = 10;
            }
        }

        //Small functions
        public float GetRandomIntBetween(int x, int y, Random rnd)
        {
            return (float)(rnd.NextDouble() * (y - x) + x);
        }

        public float Clamp01(float x)
        {
            if (x > 1)
                return 1;
            else if (x < 0)
                return 0;
            else
                return x;
        }

        public float Map(float min, float max, float x)
        {
            return x * (max - min) + min;
        }

        /// <summary>
        /// Program entry
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            Program myProgram = new Program();
            myProgram.Start();
        }

        /// <summary>
        /// Main logic and UI
        /// </summary>
        public void Start()
        {
            convolution = new Convolution();

            //Startup UI
            Console.Title = ("Convolutional Neural Network By Thomas");
            Console.WriteLine("### Convolutional Neural Network By Thomas ###");
            Console.WriteLine();
            Console.WriteLine("Loading Settings...");
            if (System.IO.File.Exists("settings.json"))
            {
                try
                {
                    string json = System.IO.File.ReadAllText("settings.json");
                    settings = JsonConvert.DeserializeObject<Settings>(json);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    //return;
                }
            } else
            {
                try
                {
                    System.IO.File.WriteAllText("settings.json", JsonConvert.SerializeObject(settings));
                } catch(Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }
            Console.WriteLine("Enter previous network path, leave bank to continue: ");
            string inPath = Console.ReadLine();
            if (inPath.Length > 0)
            {
                try
                {
                    string json = System.IO.File.ReadAllText(inPath);
                    //Use a custom json importer because JSONConvert won't work for this
                    LoadJSON(json);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }
            /*Console.WriteLine("Enter training data path, leave bank to continue: ");
            string inPath = Console.ReadLine();
            if (inPath.Length < 0)
            {
                StartConvolution();
            }
            Console.WriteLine("Enter target training data path, leave bank to continue: ");
            string targetPath = Console.ReadLine();
            if (inPath.Length > 0)
            {*/
            Console.WriteLine("Loading training data from 'TestImgs' and 'TargetImgs'...");
            convolution.LoadImages("TestImgs", "TargetImgs");//inPath, targetPath);
            //}

            Console.WriteLine("Setting up network...");
            convolution.InitialiseKernel(settings.nodesPL, settings.nodesPL, settings.nodeLayers);
            Console.WriteLine("Network set up as " + settings.nodesPL + "x" + settings.nodesPL + " kernel with " + settings.nodeLayers + " layers.");

            Console.CancelKeyPress += Console_CancelKeyPress;
            startTime = DateTime.Now;

            TrainNetwork();

            //I know I'm doing this wrong
            InterruptMenu();
            //Console.WriteLine("Press any key to exit...");
            //Console.ReadKey();
        }

        //Theoretically working, but I'm to scared to test it...
        private void LoadJSON(string json)
        {
            Parents newParents = new Parents();
            if(!json.Contains("parentKernels"))
            {
                Console.WriteLine("Invalid file!");
                return;
            }
            json = json.Remove(0, 21);//Remove head
            json = json.Remove(json.Length-5);//Remove tail

            string[] jsonSplit = json.Split('[');
            int x = 0;
            while(true)
            {
                if(jsonSplit[x] == "\n")
                {
                    //Each parent
                    int y = 0;
                    while(true)
                    {
                        List<Mat> nKernels = new List<Mat>();
                        if (jsonSplit[x+y] == "\n")
                        {
                            //Each kernel
                            List<float> newKernel = new List<float>();
                            int z = 0;
                            while(true)
                            {
                                if (jsonSplit[x + y + z].EndsWith("]\n],"))
                                    break;
                                string[] numbers = jsonSplit[x + y + z].Split(", ".ToArray());

                                //Each pixel
                                foreach (string num in numbers)
                                {
                                    try
                                    {
                                        newKernel.Add(float.Parse(num));
                                    }
                                    catch
                                    {
                                        Console.WriteLine("Error at x: " + x + ", y: " + y  + ", z: " + z + "; Not a number");
                                    }
                                }
                            }
                            int sqrtZ = (int)Math.Sqrt(z+1);
                            nKernels.Add(new Mat(sqrtZ, sqrtZ, MatType.CV_32FC1, newKernel.ToArray()));
                        } else
                        {
                            break;
                        }
                        newParents.parentKernels.Add(new Parents.Parent(0, nKernels.ToArray(), "imported"));
                        y++;
                    }
                } else
                {
                    break;
                }
                x++;
            }

            Console.WriteLine(json);
            Console.ReadKey();
        }

        private void Console_CancelKeyPress(object sender, ConsoleCancelEventArgs e)
        {
            e.Cancel = true;
            shouldPause = true;
            //InterruptMenu();
        }

        public void TrainNetwork()
        {
            Console.WriteLine("Training network...");
            Console.WriteLine();

            Random rnd = new Random();
            Parents potentialParents = new Parents();
            while (true)
            {
                if (shouldPause)
                    return;

                bool lastSuccess = false;
                int currParentInd = 0;

                if (currentParents.parentKernels.Count > 0)
                {
                    //if(currentParents.parentKernels.Length == convolution.kernels.Length)
                    //    convolution.SetKernels(currentParents.parentKernels);
                    currParentInd = (int)(rnd.NextDouble() * currentParents.parentKernels.Count);
                    convolution.SetKernels(currentParents.parentKernels[currParentInd].kernels);
                }

                convolution.MutateKernel(settings.mutationRate);

                float fitness = TestConnections(convolution, rnd);
                runningAvg.Enqueue(fitness);

                potentialParents.parentKernels.Add(new Parents.Parent(fitness, convolution.DeepCopyKernels(), gens>0?currentParents.parentKernels[currParentInd].species + " " + (iters-gens*settings.childrenPerGen):iters.ToString()));

                //Genetic Selection
                if (iters % settings.childrenPerGen == settings.childrenPerGen -1)
                {
                    currentParents = new Parents();

                    potentialParents.parentKernels.Sort(delegate (Parents.Parent x, Parents.Parent y)
                    {
                        return x.fitness.CompareTo(y.fitness);
                    });

                    for (int x = 0; x < settings.parentsPerGeneration; x++)
                    {
                        //float rndval = (float)rnd.NextDouble();
                        //float newTarget = Map(potentialParents.parentKernels.Last().fitness, potentialParents.parentKernels.First().fitness, rndval);
                        //Parents.Parent newParent = potentialParents.parentKernels.First(y => y.fitness>=newTarget);
                        Parents.Parent newParent = potentialParents.parentKernels[potentialParents.parentKernels.Count-x-1];
                        currentParents.parentKernels.Add(newParent);
                        //potentialParents.parentKernels.Remove(newParent);
                    }

                    potentialParents.parentKernels.Clear();

                    //Retest the previous best//This is tricky now, just assume all parents die after reproducing
                    /*if (currentParents.parentKernels.Length == convolution.kernels.Length)
                    {
                        convolution.SetKernels(currentParents.parentKernels);
                        potentialParents.parentFitness = TestConnections(convolution, rnd);
                    }*/

                    //TEMP
                    try
                    {
                        System.IO.File.WriteAllBytes("genOut.png", convolution.dstImg.ToBytes());
                        System.IO.File.WriteAllBytes("genIn.png", convolution.srcImgs[lastIndex].ToBytes());
                        //System.IO.File.WriteAllText(inpath, JsonConvert.SerializeObject(networkConnections));
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }

                    gens++;
                }

                if (runningAvg.Count > settings.overrallFitIters)
                    runningAvg.Dequeue();

                //GUI
                if (iters % 2 == 0)
                {
                    float overallFit = 0;
                    foreach (float i in runningAvg)
                    {
                        overallFit += i;
                    }
                    overallFit /= runningAvg.Count;

                    Console.Write("[" + (DateTime.Now-startTime).ToString(@"h\:mm\:ss") + "] Training... Current fitness: " + Math.Round(fitness)
                        + "%; Overall fitness: " + Math.Round(overallFit)
                        + "% \n[" + (DateTime.Now - startTime).ToString(@"h\:mm\:ss") + "] Best fitness (gen): " + Math.Round(potentialParents.parentKernels.FirstOrDefault().fitness)
                        + "%; Iterations: " + iters
                        + "; Generations: " + gens + "   "
                        + "\n[" + (DateTime.Now - startTime).ToString(@"h\:mm\:ss") + "] List Parents: " + "                   \n");
                    for(int i = 0; i< currentParents.parentKernels.Count;i++)
                        Console.WriteLine("Parent: " + currentParents.parentKernels[i].species.Substring(Math.Max(currentParents.parentKernels[i].species.Length-20, 0)) + " Fitness: " + currentParents.parentKernels[i].fitness + "  ");
                    Console.SetCursorPosition(0, Console.CursorTop - (3+currentParents.parentKernels.Count));
                }

                if (lastSuccess)
                    successIters++;

                iters++;
            }
        }

        public float TestConnections(Convolution convolution, Random rnd)
        {
            float avgFitness = 0;
            for (int i = 0; i < settings.fitnessAverageIters; i++)
            {
                lastIndex = (int)(rnd.NextDouble() * (convolution.srcImgs.Count - 1));
                avgFitness += convolution.EvaluateFitness(lastIndex, settings);
            }
            GC.Collect();

            avgFitness /= settings.fitnessAverageIters;
            return avgFitness;
        }

        /// <summary>
        /// Small menu which appears when training is interrupted
        /// </summary>
        public void InterruptMenu()
        {
            Console.WriteLine();
            for(int i = 0; i<settings.parentsPerGeneration+3; i++)
                Console.WriteLine();
            Console.WriteLine("## Training Interrupted!");
            Console.WriteLine("1. Evaluate image in current state. \n2. Save best weights. \n3. Reset weights. \n4. Anneal now. \n5. Display all weights. \n6. Set current mutation rate \n0. Continue training.");
            string inp = Console.ReadLine();
            switch (inp)
            {
                case "1":
                    Console.WriteLine("Enter image path to evaluate: ");
                    string evstr = Console.ReadLine();
                    convolution.LoadImage(evstr);
                    convolution.Convolve(0, settings);
                    System.IO.File.WriteAllBytes("tmpimg.png", convolution.dstImg.ToBytes());
                    Console.WriteLine("Saved output!");
                    Console.WriteLine("Press any key to continue...");
                    Console.ReadKey();
                    break;
                case "2":
                    Console.WriteLine("Save as: ");
                    string inpath = Console.ReadLine();
                    try
                    {
                        //System.IO.File.WriteAllBytes("tmpimg.png", convolution.dstImg.ToBytes());
                        //FATAL ERROR, Do not use the following:
                        //var settings = new JsonSerializerSettings { ContractResolver = new OptOutContractResolver() };
                        /*try
                        {
                            System.IO.File.WriteAllText(inpath, JsonConvert.SerializeObject(currentParents, Formatting.Indented));
                        } catch(Exception e)
                        {
                            Console.WriteLine(e.Message);
                        }*/
                        List<string> outjson = new List<string>
                        {
                            "{",
                            "\"parentKernels\":",
                            "[",
                            "[",
                            "[",
                            ""
                        };
                        for (int x = 0; x < currentParents.parentKernels.Count; x++)
                        {
                            for (int y = 0; y < settings.nodeLayers; y++)
                            {
                                for (int z = 0; z < convolution.kernels[y].Rows; z++)
                                {
                                    for (int w = 0; w < convolution.kernels[y].Cols; w++)
                                    {
                                        outjson[outjson.Count - 1] += currentParents.parentKernels[z].kernels[y].At<float>(z, w);//convolution.kernels[i].At<float>(x, j);// + ((x<convolution.kernels[i].Rows-1)&& (j < convolution.kernels[i].Cols-1)?", ":"");
                                        if (new Size(z + 1, w + 1) != convolution.kernels[y].Size())
                                            outjson[outjson.Count - 1] += ", ";
                                    }
                                    outjson.Add("");
                                    //Console.WriteLine();
                                }
                                if (y < settings.nodeLayers - 1)
                                {
                                    outjson[outjson.Count - 1] += "], ";
                                    outjson.Add("[");
                                } else
                                    outjson[outjson.Count - 1] += "]";
                                outjson.Add("");
                            }
                            if (x < currentParents.parentKernels.Count - 1)
                            {
                                outjson[outjson.Count - 1] += "], ";
                                outjson.Add("[");
                                outjson.Add("[");
                            }
                            else
                                outjson[outjson.Count - 1] += "]";
                            outjson.Add("");
                        }
                        outjson[outjson.Count - 1] += "]";
                        outjson.Add("}");
                        System.IO.File.WriteAllLines(inpath,outjson.ToArray());
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }
                    Console.WriteLine("Saved!");
                    break;
                case "3":
                    //InitialiseNetwork();
                    shouldPause = false;
                    //TrainNetwork();
                    break;
                case "4":
                    Console.WriteLine("Anneal strength as a decimal: ");
                    string inpower = Console.ReadLine();
                    float newAnneal = 0;
                    float.TryParse(inpower, out newAnneal);
                    Console.WriteLine("Annealing network for 1 iteration!");
                    //FillMutations(currAnneal);
                    Mat[] tmpMut = convolution.MutateKernel(currAnneal);
                    for (int i = 0; i < convolution.kernels.Length; i++)
                    {
                        //convolution.SetKernels();
                        Cv2.Add(convolution.kernels[i], tmpMut[i], convolution.kernels[i]);
                    }
                    break;
                case "5":
                    try
                    {
                        for (int i = 0; i < settings.nodeLayers; i++)
                        {
                            System.IO.File.WriteAllBytes("weights" + i + ".png", convolution.kernels[i].ToBytes());

                            for (int x = 0; x < convolution.kernels[i].Rows; x++)
                            {
                                for (int j = 0; j < convolution.kernels[i].Cols; j++)
                                {
                                    Console.Write(convolution.kernels[i].At<float>(x, j) + " , ");
                                }
                                Console.WriteLine();
                            }
                        }
                        //System.IO.File.WriteAllText(inpath, JsonConvert.SerializeObject(networkConnections));
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }
                    Console.WriteLine("Saved!");
                    break;
                case "6":
                    Console.WriteLine("New mutation rate as a decimal: ");
                    string inrate = Console.ReadLine();
                    try
                    {
                        settings.mutationRate = float.Parse(inrate);
                    } catch
                    {
                        Console.WriteLine("Not a float!");
                    }
                    break;
                case "0":
                    shouldPause = false;
                    TrainNetwork();
                    break;
            }
            InterruptMenu();
        }
    }

    public class OptOutContractResolver : DefaultContractResolver
    {
        protected override IList<JsonProperty> CreateProperties(Type type, MemberSerialization memberSerialization)
        {
            return base.CreateProperties(type, MemberSerialization.OptOut);
        }
    }
}
