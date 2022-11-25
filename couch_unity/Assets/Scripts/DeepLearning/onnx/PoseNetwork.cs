using Unity.Barracuda;
using System;
using System.IO;
using System.Linq;

namespace DeepLearning {
    [System.Serializable]
    public class PoseNetwork : NN {

        public NNModel Model = null;
        public WorkerFactory.Device Device = WorkerFactory.Device.GPU;

        private Instance _Instance_;

        public class Instance : Inference {
            public Tensor X = null;
            public Tensor Y = null;
            public IWorker Engine = null;

            public Tensor X_mean = null;
            public Tensor X_std = null;
            public Tensor Y_mean = null;
            public Tensor y_std = null;

            public byte[] X_mean_byte = null;
            public byte[] X_std_byte = null;
            public byte[] Y_mean_byte = null;
            public byte[] Y_std_byte = null;

            public float[] X_mean_array = null;
            public float[] X_std_array = null;
            public float[] Y_mean_array = null;
            public float[] Y_std_array = null;




            public Instance(NNModel model, WorkerFactory.Device device) {
                Model nn = ModelLoader.Load(model, false, false);
                Engine = nn.CreateWorker(device);
                X = new Tensor(nn.inputs.First().shape);


                // // X_mean = new Tensor(1, 1, 1,  );


                // X_mean_array = X_mean_byte.Select(b => (float)Convert.ToDouble(b)).ToArray();
                // X_std_array = X_std_byte.Select(b => (float)Convert.ToDouble(b)).ToArray();
                // Y_mean_array = Y_mean_byte.Select(b => (float)Convert.ToDouble(b)).ToArray();
                // Y_std_array = Y_std_byte.Select(b => (float)Convert.ToDouble(b)).ToArray();


                // Debug.Log(nn.inputs.First().shape);
                Y = new Tensor();
                Y = null;
            }
            public override int GetFeedSize() {
                return X.length;
            }

            public override Tensor GetFeed() {
                return X;
            }
            public override int GetReadSize() {
                return Y.length;
            }


            // public override float[] GetXMean() {
            //     return X_mean_array;
            // }            


            public override void Feed(float value) {
                X[0, 0, 0, Pivot] = value;
            }
            public override float Read() {
                return Y[0, 0, 0, Pivot];
            }
            public override void Predict() {
                //Multiple Inputs
                // Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
                // inputs.Add("Input1", new Tensor(new int[]{1,1,1,dim}, X.Flatten()));
                // inputs.Add("Input2", new Tensor(new int[]{1,1,1,dim}, new float[]{1.0f}));

                //Single Input
                Y = Engine.Execute(X).PeekOutput();
            }

            private float[] ReadBinary(string fn, int size) {
                if(File.Exists(fn)) {
                    float[] buffer = new float[size];
                    BinaryReader reader = new BinaryReader(File.Open(fn, FileMode.Open));
                    for(int i=0; i<size; i++) {
                        buffer[i] = reader.ReadSingle();
                    }
                    reader.Close();
                    return buffer;
                }
                else return null;
            }
        }

        public override void Create() {
            Session = _Instance_ = new Instance(Model, Device);
        }
        
        public override void Close() {
            _Instance_.X.Dispose();
            _Instance_.Y.Dispose();
            _Instance_.Engine.Dispose();
        }
     
    }
    
}